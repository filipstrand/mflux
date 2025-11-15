import math

import mlx.core as mx
from mlx import nn

from mflux.models.flux.model.flux_transformer.timestep_embedder import TimestepEmbedder


class BriaFiboTimesteps(nn.Module):
    """
    MLX port of diffusers.models.transformers.transformer_bria_fibo.BriaFiboTimesteps
    using the same get_timestep_embedding math.
    """

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
        """
        Args:
            timesteps: 1D array of shape (N,) with scalar timesteps.

        Returns:
            Array of shape (N, num_channels) with sinusoidal embeddings.
        """
        assert timesteps.ndim == 1, "Timesteps should be a 1D array"

        half_dim = self.num_channels // 2
        exponent = -math.log(self.time_theta) * mx.arange(0, half_dim, dtype=mx.float32)
        exponent = exponent / (half_dim - self.downscale_freq_shift)

        emb = mx.exp(exponent)
        emb = mx.expand_dims(timesteps.astype(mx.float32), axis=-1) * mx.expand_dims(emb, axis=0)

        # scale embeddings
        emb = self.scale * emb

        # concat sine and cosine embeddings
        sin = mx.sin(emb)
        cos = mx.cos(emb)
        emb = mx.concatenate([sin, cos], axis=-1)

        # flip sine and cosine embeddings
        if self.flip_sin_to_cos:
            emb = mx.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)

        # zero pad if odd
        if self.num_channels % 2 == 1:
            pad = mx.zeros((emb.shape[0], 1), dtype=emb.dtype)
            emb = mx.concatenate([emb, pad], axis=-1)

        return emb


class BriaFiboTimestepProjEmbeddings(nn.Module):
    """
    MLX port of diffusers.models.transformers.transformer_bria_fibo.BriaFiboTimestepProjEmbeddings.
    """

    def __init__(self, embedding_dim: int, time_theta: int = 10000):
        super().__init__()

        # Match the PyTorch reference: 256-dim sinusoidal → TimestepEmbedding MLP → embedding_dim.
        self.time_proj = BriaFiboTimesteps(
            num_channels=256,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
            time_theta=time_theta,
            scale=1.0,
        )
        self.timestep_embedder = TimestepEmbedder()

    def __call__(self, timestep: mx.array, dtype) -> mx.array:
        """
        Args:
            timestep: 1D array of shape (N,) with scalar timesteps.
            dtype: target dtype for the output (matches hidden_states.dtype).
        """
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.astype(dtype))
        return timesteps_emb
