import mlx.core as mx
import mlx.nn as nn

from mflux.models.depth_pro.dino_v2.patch_embed import PatchEmbed
from mflux.models.depth_pro.dino_v2.transformer_block import TransformerBlock


class DinoVisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_token = mx.random.normal(shape=(1, 1, 1024))
        self.pos_embed = mx.random.normal(shape=(1, 577, 1024))
        self.patch_embed = PatchEmbed()
        self.blocks = [TransformerBlock() for i in range(24)]
        self.norm = nn.LayerNorm(dims=1024, eps=1e-6)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x

    def _pos_embed(self, x: mx.array) -> mx.array:
        x = x.reshape((35, -1, 1024))
        to_cat = [mx.broadcast_to(self.cls_token, (35,) + self.cls_token.shape[1:])]
        x = mx.concatenate(to_cat + [x], axis=1)
        x = x + self.pos_embed
        return x
