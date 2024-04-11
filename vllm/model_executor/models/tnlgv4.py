from typing import List, Optional, Tuple
import math

import torch
from torch import nn

from vllm.attention import Attention, AttentionMetadata
from vllm.core.block_manager_v1 import BlockSpaceManagerV1
from vllm.core.scheduler import global_block_manager
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear,
                                               ColumnParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding, DEFAULT_VOCAB_PADDING_SIZE)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)
from vllm.sequence import SamplerOutput
from vllm.transformers_utils.configs import TLGv4Config
# from .triton_flash_blocksparse_attn import get_local_strided_sparse_attention_op, BlockSparseParams

@torch.jit.script
def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)


@torch.jit.script
def gegelu(input, limit: Optional[float] = None):
    a_gelu, a_linear = input[..., ::2], input[..., 1::2]
    if limit is not None:
        a_gelu = torch.where(
            torch.isinf(a_gelu), a_gelu, a_gelu.clamp(min=None, max=limit)
        )
        a_linear = torch.where(
            torch.isinf(a_linear), a_linear, a_linear.clamp(min=-limit, max=limit)
        )
    out_gelu = quick_gelu(a_gelu)
    return out_gelu * (a_linear + 1)


class TLGv4MLP(nn.Module):
    def __init__(
        self,
        config: TLGv4Config,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.config = config
        assert self.config.hidden_act == "gegelu", "Only `gegelu` is supported for the 4.7 series of models .."
        self.hidden_size = config.hidden_size
        self.gegelu_limit = config.gegelu_limit
        self.intermediate_size = config.intermediate_size

        # self.up_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size)
        self.up_proj = MergedColumnParallelLinear(
            self.hidden_size,
            2* [self.intermediate_size],
            bias=True,
            linear_method=linear_method,
        )
        # self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size)
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=True,
            linear_method=linear_method,
        )
        # self.dropout = nn.Dropout(config.ffn_dropout_prob)

    def forward(self, x):
        gate_up, _ = self.up_proj(x)
        x = gegelu(gate_up)
        x, _ = self.down_proj(x)
        return x
        
class TLGv4SelfAttention(nn.Module):
    def __init__(self, config: TLGv4Config, layer_idx: Optional[int] = None) -> None:
        super().__init__()
        self.config = config
        
        self.hidden_size = config.hidden_size
        # Number of Query Heads
        self.num_heads = config.num_attention_heads
        
        self.head_dim = self.hidden_size // self.num_heads
        self.tp_size = get_tensor_model_parallel_world_size()
        # Number of total Key Value Heads before tensor parallel
        self.num_key_value_heads = config.num_key_value_heads
        self.num_q_per_kv = self.num_heads // self.num_key_value_heads
        if self.num_key_value_heads >= self.tp_size:
            assert self.tp_size % self.num_key_value_heads
        self.num_kv_heads = max(1, self.num_key_value_heads // self.tp_size)
        
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_embedding_base = config.rope_embedding_base
        self.rope_position_scale = config.rope_position_scale
        self.is_causal = True

        norm_factor = None
        if config.mup_use_scaling:
            norm_factor = self.head_dim / config.mup_attn_multiplier
        else:
            norm_factor = math.sqrt(self.head_dim)

        self.query_key_value = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_key_value_heads,
            bias=True,
            linear_method=None,
        )
        self.dense = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            linear_method=None
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_embedding_base,
            rope_scaling={"type":"linear","factor":self.rope_position_scale},
        )
        
        #blocksparse params
        self.blocksparse_block_size = config.blocksparse_block_size
        self.blocksparse_num_local_blocks = config.blocksparse_num_local_blocks
        self.sliding_window = self.blocksparse_block_size * self.blocksparse_num_local_blocks
        
        self.blocksparse_vert_stride = config.blocksparse_vert_stride
        
        is_sparse = True
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              norm_factor,
                              num_kv_heads=self.num_kv_heads,
                              sliding_window=self.sliding_window,
                              selector_conf={"is_msft_backend": False, "is_sparse": is_sparse})
        is_sparse is True and self.attn.impl.build_sparse_op(config)

    def set_global_block_manager(self):
        self.block_manager = global_block_manager
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """_summary_
        """
        qkv, _ = self.query_key_value(hidden_states)
        qkv = qkv.reshape(qkv.shape[:-1]+(-1, (2+self.num_q_per_kv),self.head_dim))
        q, k, v = qkv.split([self.num_q_per_kv, 1, 1], dim=-2)
        q = q.reshape(q.shape[:2]+ (-1,))
        k = k.reshape(q.shape[:2]+ (-1,))
        v = v.reshape(q.shape[:2]+ (-1,))
        q = q.reshape(-1, self.head_dim*self.num_heads)
        k = k.reshape(-1, self.head_dim*self.num_kv_heads)
        v = v.reshape(-1, self.head_dim*self.num_kv_heads)
        q, k = self.rotary_emb(positions, q, k)
        
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata=attn_metadata)
        
        #self.block_manager.evict()
        output, _ = self.dense(attn_output)
        return output

class TLGv4DecoderLayer(nn.Module):
    def __init__(self, config: TLGv4Config, layer_idx: int, linear_method: Optional[LinearMethodBase] = None,):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = TLGv4SelfAttention(config, layer_idx)
        self.mlp = TLGv4MLP(config)

        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
    
    
class TLGv4Model(nn.Module):

    def __init__(self, config:TLGv4Config,linear_method: Optional[LinearMethodBase] = None,):
        super().__init__()
        self.config = config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, 
            config.hidden_size
            )


        self.embedding_dropout = nn.Dropout(config.embedding_dropout_prob)
        
        self.mup_embedding_multiplier = config.mup_embedding_multiplier

        self.layers = nn.ModuleList([TLGv4DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])

        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: Optional[torch.LongTensor],
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata = None,
    ):
        hidden_states = self.embed_tokens(input_ids)
        if self.mup_embedding_multiplier is not None and self.mup_embedding_multiplier > 0.0:
            hidden_states = hidden_states * self.mup_embedding_multiplier
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(
                positions,
                hidden_states,
                kv_caches[i],
                attn_metadata,
            )
        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states
        

class TLGv4ForCausalLM(nn.Module):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, linear_method: Optional[LinearMethodBase] = None,):
        super().__init__()
        self.config = config
        self.model = TLGv4Model(config)
        self.vocab_size = config.vocab_size     
        self.mup_width_multiplier = config.mup_width_multiplier
        self.lm_head = ParallelLMHead(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    
    def set_global_block_manager(self):
        self.set_block_manager = global_block_manager
        return 

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head.weight, hidden_states,
                                       sampling_metadata)
        return logits  
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: Optional[torch.LongTensor],
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        output_hidden_states = self.model(input_ids=input_ids, positions=positions, kv_caches=kv_caches, attn_metadata=attn_metadata)
        output_hidden_states = output_hidden_states
        return output_hidden_states
        
    
    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits / self.mup_width_multiplier, sampling_metadata)
        return next_tokens
    
    def hello(self):
        print("++++++++++++++++hello world++++++++++++++++++")

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            if "rotary_emb.inv_freq" in name:
                continue
            if name.endswith(".bias") and name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
        self.lm_head.weight.data.copy_(self.model.embed_tokens.weight.data)