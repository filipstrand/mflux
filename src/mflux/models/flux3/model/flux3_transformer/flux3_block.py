import mlx.core as mx
from mlx import nn

from mflux.models.flux3.model.flux3_transformer.flux3_attention import Flux3Attention
from mflux.models.flux3.model.flux3_transformer.flux3_feed_forward import Flux3FeedForward


class Flux3Block(nn.Module):
    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, mlp_ratio: float = 3.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6, affine=False)
        self.attn = Flux3Attention(dim=dim, heads=num_attention_heads, dim_head=attention_head_dim)
        self.ff = Flux3FeedForward(dim=dim, mlp_ratio=mlp_ratio)

    def __call__(
        self,
        x: mx.array,
        rotary_emb: tuple[mx.array, mx.array],
        mod: tuple[mx.array, mx.array, mx.array],
    ) -> mx.array:
        shift, scale, gate = mod
        x_mod = (1 + scale) * self.norm(x) + shift
        return x + gate * (self.attn(x_mod, rotary_emb) + self.ff(x_mod))
