import mlx.core as mx
import mlx.nn as nn

from mflux.models.depth_pro.model.dino_v2.attention import Attention
from mflux.models.depth_pro.model.dino_v2.layer_scale import LayerScale
from mflux.models.depth_pro.model.dino_v2.mlp import MLP


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(dims=1024, eps=1e-6, bias=True)
        self.attn = Attention()
        self.ls1 = LayerScale(dims=1024, init_values=1e-5)
        self.norm2 = nn.LayerNorm(dims=1024, eps=1e-6, bias=True)
        self.mlp = MLP()
        self.ls2 = LayerScale(dims=1024, init_values=1e-5)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x
