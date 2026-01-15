import math

import mlx.core as mx
from mlx import nn


class QwenTimesteps(nn.Module):
    def __init__(self, proj_dim: int = 256, scale: float = 1000.0):
        super().__init__()
        self.proj_dim = proj_dim
        self.flip_sin_to_cos = True
        self.downscale_freq_shift = 0.0
        self.scale = scale

    def __call__(self, timesteps: mx.array) -> mx.array:
        # CRITICAL FIX: Replace assert with explicit validation (disabled with -O flag)
        if len(timesteps.shape) != 1:
            raise ValueError(f"Timesteps should be a 1d-array, got shape {timesteps.shape}")

        half_dim = self.proj_dim // 2
        max_period = 10000

        # CRITICAL FIX: Validate denominator before division to prevent division by zero
        denominator = half_dim - self.downscale_freq_shift
        if abs(float(denominator)) < 1e-8:
            raise ValueError(
                f"Invalid configuration: half_dim ({half_dim}) - downscale_freq_shift "
                f"({self.downscale_freq_shift}) results in zero division"
            )

        exponent = -math.log(max_period) * mx.arange(0, half_dim, dtype=mx.float32)
        exponent = exponent / denominator

        timesteps_dtype = timesteps.dtype
        emb = mx.exp(exponent).astype(timesteps_dtype)

        emb = timesteps[:, None].astype(mx.float32) * emb[None, :]
        emb = self.scale * emb
        emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)

        if self.flip_sin_to_cos:
            emb = mx.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)

        return emb
