"""Attention layer with BlockSparse and PagedAttention."""
import importlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type

import torch
from .triton_flash_blocksparse_attn import get_local_strided_sparse_attention_op, BlockSparseParams  # noqa: E501

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata)
from vllm.attention.ops.paged_attn import (PagedAttention,
                                           PagedAttentionMetadata)
from vllm.logger import init_logger
from vllm.utils import is_hip

logger = init_logger(__name__)


class BlockSparseBackend(AttentionBackend):

    @staticmethod
    def get_impl_cls() -> Type["BlockSparseImpl"]:
        return BlockSparseImpl

    @staticmethod
    def make_metadata(*args, **kwargs) -> "BlockSparseMetadata":
        return BlockSparseMetadata(*args, **kwargs)

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return PagedAttention.get_kv_cache_shape(num_blocks, block_size,
                                                 num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: Dict[int, int],
    ) -> None:
        PagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: Dict[int, List[int]],
    ) -> None:
        PagedAttention.copy_blocks(kv_caches, src_to_dists)


@dataclass
class BlockSparseMetadata(AttentionMetadata, PagedAttentionMetadata):
    """Metadata for BlockSparseBackend.

    NOTE: Any python object stored here is not updated when it is
    cuda-graph replayed. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.
    """
    # Currently, input sequences can only contain all prompts
    # or all decoding. True if all sequences are prompts.
    is_prompt: bool
    # (num_tokens,). The indices of the token slots that input tokens will be
    # stored into. E.g., if `slot_mapping` is [35, 2, 17] and the block size
    # is 16, the three tokens are stored in the 3rd slot in block 2, 2nd slot
    # in block 0, and 1st slot in block 1, respectively.
    slot_mapping: torch.Tensor
    # (batch_size,). The prompt length per sequence. None if it is a decoding.
    prompt_lens: Optional[List[int]]
    # prompt_lens stored as a tensor.
    prompt_lens_tensor: Optional[torch.Tensor]
    # The number of prompt tokens. Doesn't include padding.
    num_prompt_tokens: int
    # The number of generation tokens. Doesn't include padding.
    num_generation_tokens: int

    # NOTE(sang): Definition of context_len, subquery_len, and seqlen.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seqlen ----------------------|
    #                                   |- subquery_len -|

    # WARNING(sang): context_len has different definition depending on if it is
    # prefill vs decoding. When it is prefill, it doesn't include new tokens.
    # When it is for decoding, it includes a new token.

    # Maximum subquery length in the batch.
    max_subquery_len: Optional[int]
    # FIXME: It is for flash attn.
    # Maximum prompt length in the batch.
    max_prompt_len: Optional[int]
    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    subquery_start_loc: Optional[torch.Tensor]
    # FIXME: It is for flash attn.
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor]

    # Whether or not if cuda graph is enabled.
    # Cuda-graph is currently enabled for decoding only.
    # TODO(woosuk): Move `use_cuda_graph` out since it's unrelated to attention.
    use_cuda_graph: bool
    attn_bias = None

    def __post_init__(self):
        # Set during the execution of the first attention op.
        # It is a list because it is needed to set per prompt
        # when alibi slopes is used. It is because of the limitation
        # from xformer API.
        # will not appear in the __repr__ and __init__
        pass

class BlockSparseImpl(AttentionImpl):
    """
    If the input tensors contain prompt tokens, the layout is as follows:
    |<--------------- num_prompt_tokens --------------->|	
    |<--prompt_0-->|<--prompt_1-->|...|<--prompt_N-1--->|

    Otherwise, the layout is as follows:	
    |<------------------ num_generation_tokens (M) ----------------->|	
    |<--generation_0-->|..........|<--generation_M-1-->|<--padding-->|

    Generation tokens can contain padding when cuda-graph is used.
    Currently, prompt tokens don't contain any padding.

    The prompts might have different lengths, while the generation tokens
    always have length 1.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        suppored_head_sizes = PagedAttention.get_supported_head_sizes()
        if head_size not in suppored_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {suppored_head_sizes}.")

        # AMD Radeon 7900 series (gfx1100) currently does not support BlockSparse
        # nor FlashAttention. As a temporary workaround, we use naive PyTorch
        # implementation of attention.
        self.use_naive_attention = _check_use_naive_attention()
        self.xformers_avaiable = importlib.util.find_spec("xformers") is not None
        self.sparse_fn = None

    def build_sparse_op(self, config):
        #print("+++++++++++++++build sparse op+++++++++++++++++++++++++++++++++")
        blocksparse_params=BlockSparseParams.from_config(config)
        self.sparse_fn = get_local_strided_sparse_attention_op(
            n_heads=self.num_heads,
            max_seq_len=config.max_position_embeddings,
            sparse_block_size=blocksparse_params.block_size,
            kernel_block_size=blocksparse_params.kernel_block_size,
            local_blocks=blocksparse_params.num_local_blocks,
            vert_stride=blocksparse_params.vert_stride,
            homo_head=blocksparse_params.homo_head_pattern,
            device="cuda",
            inference=True,
            dtype=torch.get_default_dtype(),
        )

    @staticmethod
    def build_sparse_mask_for_decode(sparse_fn, num_heads,
                                    attn_metadata: BlockSparseMetadata,
                                    kv_block_size):
        if (attn_metadata.is_prompt is True or sparse_fn is None or
            sparse_fn.local_tokens >= attn_metadata.max_context_len or
            attn_metadata.block_tables.dim() == 3):
            return
        print("++++++++++++build mask+++++++++++++++++++++++++++++++++")
        sparse_pattern = sparse_fn.sparse_pattern.cpu()
        max_context_block = 0
        sparse_block_tables = []
        sparse_context_lens = []
        block_tables = attn_metadata.block_tables.cpu()
        context_lens = attn_metadata.context_lens.cpu()
        num_kv_block_in_sparse_block = sparse_fn.sparse_block_size//kv_block_size
        for seq_id in range(block_tables.shape[0]):
            sparse_block_tables.append([])
            sparse_context_lens.append([])
            if context_lens[seq_id] < sparse_fn.local_tokens:
                valid_context_blocks = (
                    (context_lens[seq_id]+kv_block_size-1)//kv_block_size)
                sparse_block_tables[-1] = block_tables[seq_id][:valid_context_blocks].unsqueeze(
                    0).expand(num_heads, -1).tolist()
                sparse_context_lens[-1] = [context_lens[seq_id].item()
                                            for _ in range(num_heads)]
                max_context_block = max(
                    max_context_block, valid_context_blocks)
                continue
            sparse_block_id = context_lens[seq_id] // sparse_fn.sparse_block_size
            for head_id in range(num_heads):
                sparse_block_tables[-1].append([])
                sparse_mask = sparse_pattern[head_id, sparse_block_id]
                for super_block_id in range(0, block_tables.shape[-1], num_kv_block_in_sparse_block):
                    if sparse_mask[super_block_id//num_kv_block_in_sparse_block] == 1:
                        sparse_block_tables[-1][-1] += block_tables[seq_id,
                                                                    super_block_id:super_block_id+num_kv_block_in_sparse_block].tolist()
                bank_count = ((kv_block_size-context_lens[seq_id] % kv_block_size) %
                                kv_block_size)*sparse_mask[super_block_id//num_kv_block_in_sparse_block-1]
                sparse_context_lens[-1].append(
                    len(sparse_block_tables[-1][-1])*kv_block_size - int(bank_count))
            max_context_block = max(max_context_block, max(
                [len(sparse_block_tables[-1][i]) for i in range(num_heads)]))
            # sparse_context_lens[-1] = [len(sparse_block_tables[-1][i]) for i in range(self.num_heads)]

        attn_metadata.max_context_len = max(
            [max(contextl) for contextl in sparse_context_lens])
        for seq_id in range(block_tables.shape[0]):
            for head_id in range(num_heads):
                sparse_block_tables[seq_id][head_id] += [-1 for _ in range(
                    max_context_block - len(sparse_block_tables[seq_id][head_id]))]
        attn_metadata.block_tables = torch.tensor(
            sparse_block_tables, device=attn_metadata.block_tables.device, dtype=attn_metadata.block_tables.dtype)
        attn_metadata.context_lens = torch.tensor(
            sparse_context_lens, device=attn_metadata.block_tables.device, dtype=attn_metadata.context_lens.dtype)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: BlockSparseMetadata,
        kv_scale: float,
    ) -> torch.Tensor:
        """Forward pass with BlockSparse and PagedAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        num_tokens, hidden_size = query.shape
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        if kv_cache is not None:
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size)

            # Reshape the input keys and values and store them in the cache.
            # If kv_cache is not provided, the new key and value tensors are
            # not cached. This happens during the initial memory profiling run.
            PagedAttention.write_to_paged_cache(key, value, key_cache,
                                                value_cache,
                                                attn_metadata.slot_mapping,
                                                attn_metadata.kv_cache_dtype,
                                                kv_scale)

        if attn_metadata.is_prompt:
            # Prompt run.
            if kv_cache is None or attn_metadata.block_tables.numel() == 0:
                # normal attention.
                # block tables are empty if the prompt does not have a cached
                # prefix.
                if self.num_kv_heads != self.num_heads:
                    # As of Nov 2023, BlockSparse only supports MHA. For MQA/GQA,
                    # project the key and value tensors to the desired number of
                    # heads.
                    # TODO(woosuk): Use MQA/GQA kernels for higher performance.
                    query = query.view(query.shape[0], self.num_kv_heads,
                                       self.num_queries_per_kv,
                                       query.shape[-1])
                    key = key[:, :,
                              None, :].expand(key.shape[0], self.num_kv_heads,
                                              self.num_queries_per_kv,
                                              key.shape[-1])
                    value = value[:, :,
                                  None, :].expand(value.shape[0],
                                                  self.num_kv_heads,
                                                  self.num_queries_per_kv,
                                                  value.shape[-1])
                if self.xformers_avaiable and (attn_metadata.max_prompt_len is None or attn_metadata.max_prompt_len <= self.sparse_fn.local_tokens):
                    output = self._run_memory_efficient_xformers_forward(query, key, value, attn_metadata)
                elif self.use_naive_attention:
                    query = query.view(-1, self.num_heads, self.head_size)
                    key = key.reshape(-1, self.num_heads, self.head_size)
                    value = value.reshape(-1, self.num_heads, self.head_size)
                    output = torch.empty_like(query)
                    start = 0
                    for _, prompt_len in enumerate(attn_metadata.prompt_lens):
                        end = start + prompt_len
                        out = _naive_masked_attention(
                            query[None, start:end],
                            key[None, start:end],
                            value[None, start:end],
                            self.num_heads,
                            self.num_kv_heads,
                            self.head_size,
                            self.scale,
                            self.sparse_fn.sparse_pattern,
                            block_size=self.sparse_fn.sparse_block_size
                        )
                        # TODO(woosuk): Unnecessary copy. Optimize.
                        output[start:end].copy_(out)
                        start += prompt_len

                    # Using view got RuntimeError: view size is not compatible
                    # with input tensor's size and stride (at least one
                    # dimension spans across two contiguous subspaces).
                    # Use reshape instead.
                    output = output.reshape(num_tokens, hidden_size)
                else:
                    query = query.view(-1, self.num_heads, self.head_size)
                    key = key.reshape(-1, self.num_heads, self.head_size)
                    value = value.reshape(-1, self.num_heads, self.head_size)
                    output = torch.empty_like(query)
                    start = 0
                    for _, prompt_len in enumerate(attn_metadata.prompt_lens):
                        end = start + prompt_len
                        out = self._run_BlockSparse_forward(
                                query[start:end],
                                key[start:end],
                                value[start:end],
                            attn_metadata)
                        output[start:end].copy_(out)
                        start += prompt_len

                    # Using view got RuntimeError: view size is not compatible
                    # with input tensor's size and stride (at least one
                    # dimension spans across two contiguous subspaces).
                    # Use reshape instead.
                    output = output.reshape(num_tokens, hidden_size)            
            else:
                # prefix-enabled attention
                # TODO(Hai) this triton kernel has regression issue (broke) to
                # deal with different data types between KV and FP8 KV cache,
                # to be addressed separately.
                output = PagedAttention.forward_prefix(
                    query,
                    key,
                    value,
                    key_cache,
                    value_cache,
                    attn_metadata.block_tables,
                    attn_metadata.subquery_start_loc,
                    attn_metadata.prompt_lens_tensor,
                    attn_metadata.context_lens,
                    attn_metadata.max_subquery_len,
                    self.alibi_slopes,
                )
        else:
            kv_block_size = value_cache.shape[-1]
            self.build_sparse_mask_for_decode(
                self.sparse_fn, self.num_heads, attn_metadata, kv_block_size)
            # Decoding run.
            output = PagedAttention.forward_decode(
                query,
                key_cache,
                value_cache,
                attn_metadata.block_tables,
                attn_metadata.context_lens,
                attn_metadata.max_context_len,
                attn_metadata.kv_cache_dtype,
                self.num_kv_heads,
                self.scale,
                self.alibi_slopes,
                kv_scale,
            )

        # Reshape the output tensor.
        return output.view(-1, self.num_heads * self.head_size)

    def _run_memory_efficient_xformers_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata,
    ) -> torch.Tensor:
        from xformers import ops as xops
        from xformers.ops.fmha.attn_bias import (BlockDiagonalCausalMask)
        """Attention for 1D query of multiple prompts. Multiple prompt
        tokens are flattened in to `query` input.

        Args:
            output: shape = [num_prompt_tokens, num_heads, head_size]
            query: shape = [num_prompt_tokens, num_heads, head_size]
            key: shape = [num_prompt_tokens, num_kv_heads, head_size]
            value: shape = [num_prompt_tokens, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        """
        query = query.view(query.shape[0], self.num_heads, -1)
        key = key.reshape(query.shape[0], self.num_heads, -1)
        value = value.reshape(query.shape[0], self.num_heads, -1)
        # Set attention bias if not provided. This typically happens at
        # the very attention layer of every iteration.
        # FIXME(woosuk): This is a hack.
        if attn_metadata.attn_bias is None:
            if self.alibi_slopes is None:
                attn_bias = BlockDiagonalCausalMask.from_seqlens(
                    attn_metadata.prompt_lens)
                if self.sliding_window is not None:
                    attn_bias = attn_bias.make_local_attention(
                        self.sliding_window)
                attn_metadata.attn_bias = [attn_bias]
            else:
                attn_metadata.attn_bias = _make_alibi_bias(
                    self.alibi_slopes, self.num_kv_heads, query.dtype,
                    attn_metadata.prompt_lens)

        op = xops.fmha.MemoryEfficientAttentionFlashAttentionOp[0] if (
            is_hip()) else None
        # No alibi slopes.
        # TODO(woosuk): Too many view operations. Let's try to reduce
        # them in the future for code readability.
        if self.alibi_slopes is None:
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            out = xops.memory_efficient_attention_forward(
                query,
                key,
                value,
                attn_bias=attn_metadata.attn_bias[0],
                p=0.0,
                scale=self.scale,
                op=op)

            return out.view_as(query)

        # Attention with alibi slopes.
        # FIXME(woosuk): Because xformers does not support dynamic sequence
        # lengths with custom attention bias, we process each prompt one by
        # one. This is inefficient, especially when we have many short prompts.
        output = torch.empty_like(query)
        start = 0
        for i, prompt_len in enumerate(attn_metadata.prompt_lens):
            end = start + prompt_len
            out = xops.memory_efficient_attention_forward(
                query[None, start:end],
                key[None, start:end],
                value[None, start:end],
                attn_bias=attn_metadata.attn_bias[i],
                p=0.0,
                scale=self.scale,
                op=op)
            # TODO(woosuk): Unnecessary copy. Optimize.
            output[start:end].copy_(out.squeeze(0))
            start += prompt_len
        return output

    def _run_BlockSparse_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: BlockSparseMetadata,
    ) -> torch.Tensor:
        """Attention for 1D query of multiple prompts. Multiple prompt
        tokens are flattened in to `query` input.

        Args:
            output: shape = [num_prompt_tokens, num_heads, head_size]
            query: shape = [num_prompt_tokens, num_heads, head_size]
            key: shape = [num_prompt_tokens, num_kv_heads, head_size]
            value: shape = [num_prompt_tokens, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        """
        seq_len = query.shape[0]
        query = query.view(1, seq_len, self.num_heads, -1)
        key = key.reshape(1, seq_len, self.num_heads, -1)
        value = value.reshape(1, seq_len, self.num_heads, -1)
        # No alibi slopes.
        # TODO(woosuk): Too many view operations. Let's try to reduce
        # them in the future for code readability.
        if self.alibi_slopes is None:
            query_sp = query.permute(0, 2, 1, 3).contiguous()
            # shape: (bs, nkvp, q_len, hn)
            key = key.permute(0, 2, 1, 3).contiguous()
            # shape: (bs, nkvp, q_len, hn)
            value = value.permute(0, 2, 1, 3).contiguous()
            #positions = [0] + attn_metadata.prompt_lens
            #acc_context_lens = torch.cumsum(torch.tensor(positions, device=key.device), 0),
            out = self.sparse_fn(
                    query_sp,
                    key,
                    value,
                    self.scale,
                    None,
                    attn_metadata.max_prompt_len,
                    None,
            )
            out = out.permute(0,2,1,3).contiguous()
            return out.view_as(query).squeeze(0)

        # Attention with alibi slopes.
        # FIXME(woosuk): Because BlockSparse does not support dynamic sequence
        # lengths with custom attention bias, we process each prompt one by
        # one. This is inefficient, especially when we have many short prompts.
        output = torch.empty_like(query)
        raise NotImplementedError("Alibi bias is not supported with BlockSparse.")
        return output


def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    num_kv_heads: int,
    dtype: torch.dtype,
    prompt_lens: List[int],
):
    attn_biases = []
    raise NotImplementedError("Alibi bias is not supported with BlockSparse.")
    return attn_biases


def _check_use_naive_attention() -> bool:
    if is_hip():
        return True
    # For ROCm, check whether flash attention is installed or not.
    use_naive_attention = importlib.util.find_spec("triton") is None
    if use_naive_attention:
        logger.warning("triton is not installed. Using naive attention. "
                       "This will take significantly more GPU memory.")
        return True
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0]<8:
        logger.warning("requires sm8.0+ to use trion based flash-attn. Using naive attention.")  # noqa: E501
        return True

    return False



def torch_attention(q, k, v, attn_mask=None, sm_scale=None, block_attn_mask=None, block_size=128, do=None):
    '''
    q, k, v: shape=(batch, n_heads, seq, dim)
    '''
    # for verification
    if sm_scale is None:
        sm_scale = math.sqrt(float(q.size(-1)))

    if block_attn_mask is not None:
        assert attn_mask is None
        seq_len = q.size(0)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()* sm_scale
        mask = block_attn_mask[..., : (
            seq_len // block_size + 1), : (seq_len // block_size + 1)]
        mask = torch.kron(mask, torch.ones(block_size, block_size, device=mask.device))
        mask = mask[:, :seq_len, :seq_len]
        mask.masked_fill_(
                torch.arange(0, seq_len, device=mask.device)[:, None] 
                < torch.arange(0, seq_len, device=mask.device)[None, :], 0)
        attn = attn.masked_fill((1 - mask).bool(), float('-inf'))
        attn = attn.softmax(-1)
        torch_output = torch.einsum('hqk,khd->qhd', attn.type_as(v), v)
    else:
        attn = torch.einsum('bhmd,bhnd->bhmn', q, k).float() * sm_scale
        # import ipdb; ipdb.set_trace()
        if attn_mask is not None:
            attn = attn.masked_fill((1 - attn_mask).bool(), float('-inf'))
        # print(f'> torch attn: {attn.exp().sum(-1)=}')

        attn = attn.softmax(-1)
        if do is not None:
            dv = torch.einsum('bhqk,bhqd->bhkd', attn.type_as(do), do)
            print(f'> torch_attn computed dv: {dv=}')
        torch_output = torch.einsum('bhmn,bhnd->bhmd', attn.type_as(v), v)
    return torch_output

def _naive_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    scale: float,
    block_attn_mask: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    query = query.view(-1, num_heads, head_size)
    key = key.reshape(-1, num_heads, head_size)
    value = value.reshape(-1, num_heads, head_size)
    #return torch_attention(query, key, value, sm_scale=scale, block_attn_mask=block_attn_mask, block_size=block_size)  # noqa: E501
    seq_len, _, _ = query.shape
    attn_mask = torch.triu(torch.ones(seq_len,
                                      seq_len,
                                      dtype=query.dtype,
                                      device=query.device),
                           diagonal=1)
    attn_mask = attn_mask * torch.finfo(query.dtype).min
    attn_weights = scale * torch.einsum("qhd,khd->hqk", query, key).float()
    attn_weights = attn_weights + attn_mask.float()
    attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
    out = torch.einsum("hqk,khd->qhd", attn_weights, value)
    return out
