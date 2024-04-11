"""Attention layer."""
from typing import List, Optional

import torch
import torch.nn as nn

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.attention.selector import get_attn_backend


class OnnxPagedAttention(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        num_heads,
        num_kv_heads,
        head_size,
        scale,
        query: torch.Tensor,  # [num_tokens, num_heads * head_size]
        key: torch.Tensor,  # [num_tokens, num_heads * head_size]
        value: torch.Tensor,  # [num_tokens, num_heads * head_size]
        key_cache: Optional[
            torch.
            Tensor],  # [num_blocks, num_heads, head_size/x, block_size, x]
        value_cache: Optional[
            torch.Tensor],  # [num_blocks, num_heads, head_size, block_size]
        # [num_blocks, 2, num_heads, head_size//quant_block_size, block_size]
        attn_metadata: AttentionMetadata,
        kv_scale: float = 1.0,
        is_sparse:bool=False,
    ) -> torch.Tensor:
        return query

    @staticmethod
    def symbolic(g: torch.Graph, num_heads, num_kv_heads, head_size, scale,
                 query, key, value, key_cache, value_cache,
                 attn_metadata, kv_scale:float=1.0, is_sparse: bool = False) -> torch.Value:
        return g.op("vllm.ort.ext::PagedAttention",
                    query,
                    key,
                    value,
                    key_cache,
                    value_cache,
                    attn_metadata,
                    outputs=1,
                    num_heads_i=num_heads,
                    num_kv_heads_i=num_kv_heads,
                    head_size_i=head_size,
                    scale_f=scale,
                    kv_scale_f=kv_scale,
                    is_sparse_i=int(is_sparse))


class Attention(nn.Module):
    """Attention layer.

    This class takes query, key, and value tensors as input. The input tensors
    can either contain prompt tokens or generation tokens.
    The class does the following:

    1. Store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention.
    3. Return the output tensor.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
        selector_conf:dict = {},
    ) -> None:
        super().__init__()
        self.backend = get_attn_backend(
            torch.get_default_dtype(), **selector_conf)
        impl_cls = self.backend.get_impl_cls()
        self.impl = impl_cls(num_heads, head_size, scale, num_kv_heads,
                             alibi_slopes, sliding_window)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        kv_scale: float = 1.0,
    ) -> torch.Tensor:
        if torch.onnx.is_in_onnx_export():
            key_cache, value_cache = kv_cache
            return OnnxPagedAttention().apply(self.impl.num_heads,
                                              self.impl.num_kv_heads,
                                              self.impl.head_size,
                                              self.impl.scale, query, key,
                                              value, key_cache, value_cache,
                                              attn_metadata, kv_scale, hasattr(self.impl, "sparse_fn"))
        return self.impl.forward(query, key, value, kv_cache, attn_metadata,
                                 kv_scale)
