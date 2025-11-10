import math

import mlx.core as mx
from mlx import nn


class QwenTimesteps(nn.Module):
    def __init__(self, proj_dim: int = 256, scale: float = 1000.0):
        super().__init__()
        self.proj_dim = proj_dim
        self.flip_sin_to_cos = True  # Qwen uses flip_sin_to_cos=True
        self.downscale_freq_shift = 0.0  # Qwen uses downscale_freq_shift=0
        self.scale = scale

    def __call__(self, timesteps: mx.array) -> mx.array:
        """
        Matches PyTorch get_timestep_embedding exactly (transformer_qwenimage.py:48-99).
        """
        # PyTorch: assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
        assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

        half_dim = self.proj_dim // 2
        max_period = 10000

        # PyTorch: exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = -math.log(max_period) * mx.arange(0, half_dim, dtype=mx.float32)

        # PyTorch: exponent = exponent / (half_dim - downscale_freq_shift)
        exponent = exponent / (half_dim - self.downscale_freq_shift)

        # PyTorch: emb = torch.exp(exponent).to(timesteps.dtype)
        # Convert to timesteps dtype BEFORE multiplication (matches PyTorch)
        timesteps_dtype = timesteps.dtype
        emb = mx.exp(exponent).astype(timesteps_dtype)

        # PyTorch: emb = timesteps[:, None].float() * emb[None, :]
        emb = timesteps[:, None].astype(mx.float32) * emb[None, :]

        # PyTorch: emb = scale * emb
        emb = self.scale * emb

        # PyTorch: emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)

        # PyTorch: if flip_sin_to_cos: emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
        if self.flip_sin_to_cos:
            emb = mx.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)

        return emb
