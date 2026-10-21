import math

import mlx.core as mx
from mlx import nn

from mflux.models.common.compute_precision import ComputePrecision
from mflux.models.flux3.model.flux3_transformer.flux3_pos_embed import Flux3PosEmbed


class Flux3Attention(nn.Module):
    # Attention output reaches ~3e4 in the bf16 reference run; V is taken this many times smaller and the output
    # scaled back in the stream dtype (attention is linear in V, no biases: exact up to rounding).
    FLOAT16_HEADROOM = 4.0

    def __init__(self, dim: int, heads: int, dim_head: int):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.norm_q = nn.RMSNorm(dim_head, eps=1e-6)
        self.norm_k = nn.RMSNorm(dim_head, eps=1e-6)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.compute_precision = ComputePrecision()

    def __call__(self, x: mx.array, rotary_emb: tuple[mx.array, mx.array]) -> mx.array:
        precision = self.compute_precision
        dtype = x.dtype
        x = precision.to_compute(x)
        batch, length, _ = x.shape
        split = lambda t: t.reshape(batch, length, self.heads, self.dim_head).transpose(0, 2, 1, 3)  # noqa: E731
        query, key, value = (
            split(self.to_q(x)),
            split(self.to_k(x)),
            split(precision.shrink(self.to_v(x), Flux3Attention.FLOAT16_HEADROOM)),
        )
        cos, sin = rotary_emb
        query = self._norm_rope(query, self.norm_q, cos, sin)
        key = self._norm_rope(key, self.norm_k, cos, sin)
        out = mx.fast.scaled_dot_product_attention(query, key, value, scale=1.0 / math.sqrt(self.dim_head))  # fmt: off
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, self.heads * self.dim_head)
        return precision.to_stream(self.to_out(out), dtype, Flux3Attention.FLOAT16_HEADROOM)

    @staticmethod
    def _norm_rope(x: mx.array, norm: nn.RMSNorm, cos: mx.array, sin: mx.array) -> mx.array:
        # RMSNorm and RoPE in float32, then back to the stream dtype.
        x32 = mx.fast.rms_norm(x.astype(mx.float32), norm.weight.astype(mx.float32), norm.eps)
        return Flux3PosEmbed.apply_rope(x32, cos, sin).astype(x.dtype)
