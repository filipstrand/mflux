"""
DistilledGuidanceLayer for Chroma model.

This module replaces FLUX's TimeTextEmbed with a pre-computed modulation approach.
It generates all 344 modulation vectors upfront, which are then distributed to:
- Single transformer blocks (indices 0-113): 38 blocks × 3 mods each
- Joint transformer blocks (indices 114-341): 19 blocks × (6 img + 6 txt) mods
- Final normalization (indices 342-343): 2 mods

Architecture:
1. ChromaCombinedTimestepTextProjEmbeddings: Creates timestep-indexed input
2. ChromaApproximator: 5-layer MLP with RMSNorm, outputs modulation vectors
"""

import math

import mlx.core as mx
from mlx import nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class ApproximatorBlock(nn.Module):
    """
    Single block of the ChromaApproximator.

    Equivalent to PixArtAlphaTextProjection with SiLU activation.
    Two linear layers with SiLU activation in between.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear_1(x)
        x = nn.silu(x)
        x = self.linear_2(x)
        return x


class ChromaApproximator(nn.Module):
    """
    The approximator network that converts timestep embeddings to modulations.

    Architecture:
    - in_proj: Linear(in_dim, hidden_dim)
    - 5 layers: each with RMSNorm + ApproximatorBlock (linear->silu->linear) + residual
    - out_proj: Linear(hidden_dim, out_dim)
    """

    def __init__(
        self,
        in_dim: int = 64,
        out_dim: int = 3072,
        hidden_dim: int = 5120,
        n_layers: int = 5,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = [ApproximatorBlock(hidden_dim) for _ in range(n_layers)]
        self.norms = [RMSNorm(hidden_dim) for _ in range(n_layers)]
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        # Project input to hidden dimension
        x = self.in_proj(x)

        # Apply residual blocks with pre-norm
        for layer, norm in zip(self.layers, self.norms):
            x = x + layer(norm(x))

        # Project to output dimension
        return self.out_proj(x)


def get_timestep_embedding(
    timesteps: mx.array,
    embedding_dim: int,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 0,
) -> mx.array:
    """
    Sinusoidal timestep embeddings.

    Args:
        timesteps: 1D array of timestep values
        embedding_dim: Dimension of the output embeddings
        flip_sin_to_cos: Whether to flip sin to cos
        downscale_freq_shift: Frequency shift for downscaling

    Returns:
        Timestep embeddings of shape [len(timesteps), embedding_dim]
    """
    half_dim = embedding_dim // 2
    exponent = -math.log(10000) * mx.arange(0, half_dim, dtype=mx.float32)
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = mx.exp(exponent)
    emb = mx.expand_dims(timesteps.astype(mx.float32), axis=-1) * mx.expand_dims(emb, axis=0)

    if flip_sin_to_cos:
        emb = mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=-1)
    else:
        emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)

    return emb


class DistilledGuidanceLayer(nn.Module):
    """
    Combined timestep embedding and approximator for Chroma model.

    This layer:
    1. Creates sinusoidal embeddings for timestep and modulation indices
    2. Processes them through a 5-layer MLP to produce modulation vectors
    3. Returns all 344 modulation vectors (each with inner_dim dimensions)

    Modulation distribution:
    - Indices 0-113: Single blocks (38 × 3 = 114)
    - Indices 114-227: Joint blocks image mods (19 × 6 = 114)
    - Indices 228-341: Joint blocks text mods (19 × 6 = 114)
    - Indices 342-343: Final norm mods (2)
    """

    def __init__(
        self,
        num_channels: int = 64,
        out_dim: int = 344,
        inner_dim: int = 3072,
        hidden_dim: int = 5120,
        n_layers: int = 5,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.out_dim = out_dim
        self.inner_dim = inner_dim

        # Pre-compute modulation index embeddings (constant, shape: [out_dim, num_channels*2])
        # These are sinusoidal embeddings for indices 0, 1000, 2000, ..., (out_dim-1)*1000
        mod_indices = mx.arange(out_dim) * 1000
        self._mod_proj = get_timestep_embedding(
            mod_indices,
            2 * (num_channels // 4),  # 2 * 16 = 32
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )

        # The approximator network
        self.approximator = ChromaApproximator(
            in_dim=num_channels,  # 64
            out_dim=inner_dim,  # 3072
            hidden_dim=hidden_dim,  # 5120
            n_layers=n_layers,  # 5
        )

    def __call__(self, timestep: mx.array) -> mx.array:
        """
        Generate modulation vectors for the given timestep.

        Args:
            timestep: Timestep value(s), shape [batch] or scalar

        Returns:
            Modulation vectors, shape [batch, out_dim, inner_dim]
            e.g., [batch, 344, 3072]
        """
        # Ensure timestep is 1D
        if timestep.ndim == 0:
            timestep = mx.expand_dims(timestep, axis=0)

        batch_size = timestep.shape[0]

        # Scale timestep to [0, 1000] range if not already
        # (diffusers multiplies by 1000 in forward pass)
        timestep = timestep * 1000.0

        # Create timestep embeddings, shape: [batch, num_channels//4] = [batch, 16]
        timesteps_proj = get_timestep_embedding(
            timestep,
            self.num_channels // 4,  # 16
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )

        # Create guidance embeddings (zeros for now, could add guidance scale support)
        # Shape: [batch, num_channels//4] = [batch, 16]
        guidance_proj = mx.zeros_like(timesteps_proj)

        # Get modulation index embeddings, shape: [out_dim, num_channels//2] = [344, 32]
        mod_proj = self._mod_proj

        # Repeat mod_proj for batch, shape: [batch, out_dim, 32]
        mod_proj = mx.broadcast_to(mx.expand_dims(mod_proj, axis=0), (batch_size, self.out_dim, mod_proj.shape[-1]))

        # Combine timestep and guidance, shape: [batch, 32]
        timestep_guidance = mx.concatenate([timesteps_proj, guidance_proj], axis=1)

        # Repeat for each modulation index, shape: [batch, out_dim, 32]
        timestep_guidance = mx.broadcast_to(
            mx.expand_dims(timestep_guidance, axis=1), (batch_size, self.out_dim, timestep_guidance.shape[-1])
        )

        # Concatenate to form input vector, shape: [batch, out_dim, 64]
        input_vec = mx.concatenate([timestep_guidance, mod_proj], axis=-1)

        # Pass through approximator, shape: [batch, out_dim, inner_dim]
        return self.approximator(input_vec)
