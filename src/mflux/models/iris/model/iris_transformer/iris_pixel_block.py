import mlx.core as mx
from mlx import nn

from mflux.models.iris.model.iris_transformer.iris_feed_forward import IrisMLP
from mflux.models.iris.model.iris_transformer.iris_rms_norm import IrisRMSNorm
from mflux.models.iris.model.iris_transformer.iris_self_attention import IrisSelfAttention


class IrisPixelBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        cond_dim: int,
        pixels_per_patch: int,
        attn_hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.pixels_per_patch = pixels_per_patch
        self.norm1 = IrisRMSNorm(hidden_size, eps)
        self.norm2 = IrisRMSNorm(hidden_size, eps)
        self.adaln = nn.Linear(cond_dim, 4 * hidden_size * pixels_per_patch, bias=True)
        self.compress = nn.Linear(pixels_per_patch * hidden_size, attn_hidden_size, bias=True)
        self.expand = nn.Linear(attn_hidden_size, pixels_per_patch * hidden_size, bias=True)
        self.attn = IrisSelfAttention(attn_hidden_size, num_heads, qk_norm=True, eps=eps)
        self.mlp = IrisMLP(hidden_size, int(hidden_size * mlp_ratio), activation="gelu")

    def __call__(self, x: mx.array, cond: mx.array, rope: tuple[mx.array, mx.array], grid: tuple[int, int]) -> mx.array:
        mods = self.adaln(cond).reshape(x.shape[0], self.pixels_per_patch, -1)
        scale1, shift1, scale2, shift2 = mx.split(mods, 4, axis=-1)
        attn = self._global_attn(self.norm1(x), rope, grid)
        x = x + (attn * (1 + scale1) + shift1)
        mlp = self.mlp(self.norm2(x))
        return x + (mlp * (1 + scale2) + shift2)

    def _global_attn(self, pixels: mx.array, rope: tuple[mx.array, mx.array], grid: tuple[int, int]) -> mx.array:
        n_patches = grid[0] * grid[1]
        batch = pixels.shape[0] // n_patches
        compact = self.compress(pixels.reshape(batch * n_patches, -1))
        attn = self.attn(compact.reshape(batch, n_patches, -1), rope=rope)
        return self.expand(attn.reshape(batch * n_patches, -1)).reshape(pixels.shape)
