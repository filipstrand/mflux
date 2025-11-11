import math

import mlx.core as mx
from mlx import nn


class BriaFiboTimesteps(nn.Module):
    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool,
        downscale_freq_shift: float,
        scale: float = 1.0,
        time_theta: int = 10000,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale
        self.time_theta = time_theta

    def __call__(self, timesteps: mx.array) -> mx.array:
        half_dim = self.num_channels // 2
        exponent = -math.log(self.time_theta) * mx.arange(0, half_dim, dtype=mx.float32)
        exponent = exponent / (half_dim - self.downscale_freq_shift)
        emb = mx.exp(exponent)
        emb = mx.expand_dims(timesteps.astype(mx.float32), axis=-1) * mx.expand_dims(emb, axis=0)
        emb = self.scale * emb
        sin = mx.sin(emb)
        cos = mx.cos(emb)
        emb = mx.concatenate([sin, cos], axis=-1)
        if self.flip_sin_to_cos:
            emb = mx.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)
        if self.num_channels % 2 == 1:
            pad = mx.zeros((emb.shape[0], 1), dtype=emb.dtype)
            emb = mx.concatenate([emb, pad], axis=-1)
        return emb
