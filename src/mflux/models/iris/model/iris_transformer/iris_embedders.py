import math

import mlx.core as mx
import numpy as np
from mlx import nn


class IrisTimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, freq_dim: int = 256, max_period: float = 10.0):
        super().__init__()
        self.freq_dim = freq_dim
        self.max_period = max_period
        self.linear_1 = nn.Linear(freq_dim, hidden_size, bias=True)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def __call__(self, t: mx.array, dtype: mx.Dtype) -> mx.array:
        n = self.freq_dim // 2
        k = mx.arange(n, dtype=mx.float32)
        phase = t.astype(mx.float32)[:, None] * mx.exp(-math.log(self.max_period) * k / n)[None, :]
        emb = mx.concatenate([mx.cos(phase), mx.sin(phase)], axis=-1).astype(dtype)
        return self.linear_2(nn.silu(self.linear_1(emb)))[:, None, :]


class IrisPatchEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_size, bias=True)

    def __call__(self, patches: mx.array) -> mx.array:
        return self.proj(patches)


class IrisPixelEmbedder(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_channels, hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        batch, _, height, width = x.shape
        p = self.patch_size
        tokens = self.proj(x.transpose(0, 2, 3, 1))
        tokens = tokens + self._pos_embed(height, width, self.proj.weight.shape[0]).astype(tokens.dtype)
        tokens = tokens.reshape(batch, height // p, p, width // p, p, -1).transpose(0, 1, 3, 2, 4, 5)
        return tokens.reshape(batch * (height // p) * (width // p), p * p, -1)

    @staticmethod
    def _pos_embed(height: int, width: int, embed_dim: int) -> mx.array:
        grid_w, grid_h = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        emb = np.concatenate(
            [
                IrisPixelEmbedder._sincos_1d(embed_dim // 2, grid_w),
                IrisPixelEmbedder._sincos_1d(embed_dim // 2, grid_h),
            ],
            axis=1,
        )
        return mx.array(emb.astype(np.float32)).reshape(height, width, embed_dim)

    @staticmethod
    def _sincos_1d(embed_dim: int, pos: np.ndarray) -> np.ndarray:
        omega = 1.0 / 10000.0 ** (np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0))
        out = np.outer(pos.reshape(-1).astype(np.float64), omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)
