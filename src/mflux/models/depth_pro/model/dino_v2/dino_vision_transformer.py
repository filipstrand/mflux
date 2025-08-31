import mlx.core as mx
import mlx.nn as nn

from mflux.models.depth_pro.model.dino_v2.patch_embed import PatchEmbed
from mflux.models.depth_pro.model.dino_v2.transformer_block import TransformerBlock


class DinoVisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_token = mx.random.normal(shape=(1, 1, 1024))
        self.pos_embed = mx.random.normal(shape=(1, 577, 1024))
        self.patch_embed = PatchEmbed()
        self.blocks = [TransformerBlock() for i in range(24)]
        self.norm = nn.LayerNorm(dims=1024, eps=1e-6, bias=True)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        backbone_highres_hook0 = None
        backbone_highres_hook1 = None

        x = self.patch_embed(x)
        x = self._pos_embed(x)
        for i, block in enumerate(self.blocks):
            x = block(x)

            # Save intermediary results for later
            if i == 5:
                backbone_highres_hook0 = x
            if i == 11:
                backbone_highres_hook1 = x

        x = self.norm(x)
        return x, backbone_highres_hook0, backbone_highres_hook1

    def _pos_embed(self, x: mx.array) -> mx.array:
        B, H, W, C = x.shape
        x = x.reshape((B, -1, 1024))
        to_cat = [mx.broadcast_to(self.cls_token, (B,) + self.cls_token.shape[1:])]
        x = mx.concatenate(to_cat + [x], axis=1)
        x = x + self.pos_embed
        return x
