import mlx.core as mx
import mlx.nn as nn

from mflux.models.depth_pro.dino_v2.block import Block
from mflux.models.depth_pro.dino_v2.patch_embed import PatchEmbed


class DinoVisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.pos_embed = nn.Linear(input_dims=1, output_dims=1)
        self.cls_token = nn.Linear(input_dims=1, output_dims=1)
        self.blocks = [Block() for i in range(24)]
        self.norm = nn.LayerNorm(dims=1024, eps=1e-6)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x
