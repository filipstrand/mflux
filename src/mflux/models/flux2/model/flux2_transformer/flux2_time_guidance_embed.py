"""Time and guidance embedding for FLUX.2.

FLUX.2 uses a simpler time/guidance embedding than FLUX.1:
- No text embedder (pooled text is handled separately via context_embedder)
- Only timestep and guidance embedders
- Output dimension is hidden_dim (6144), not conditioning_dim (256)
"""

import math

import mlx.core as mx
from mlx import nn

from mflux.models.common.config import ModelConfig


class TimestepEmbedder(nn.Module):
    """Embeds timestep into a vector.

    Args:
        in_dim: Input dimension from time projection (256)
        hidden_dim: Output hidden dimension (6144)
    """

    def __init__(self, in_dim: int = 256, hidden_dim: int = 6144):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def __call__(self, timestep: mx.array) -> mx.array:
        hidden_states = self.linear_1(timestep)
        hidden_states = nn.silu(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class GuidanceEmbedder(nn.Module):
    """Embeds guidance into a vector.

    Args:
        in_dim: Input dimension from time projection (256)
        hidden_dim: Output hidden dimension (6144)
    """

    def __init__(self, in_dim: int = 256, hidden_dim: int = 6144):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def __call__(self, guidance: mx.array) -> mx.array:
        hidden_states = self.linear_1(guidance)
        hidden_states = nn.silu(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class Flux2TimeGuidanceEmbed(nn.Module):
    """Time and guidance embedding for FLUX.2.

    Unlike FLUX.1's TimeTextEmbed, this does not include a text embedder.
    The pooled text embedding is handled via the context_embedder in the transformer.

    Args:
        in_dim: Input dimension from sinusoidal projection (256)
        hidden_dim: Output hidden dimension (6144)
    """

    def __init__(self, in_dim: int = 256, hidden_dim: int = 6144):
        super().__init__()
        self.timestep_embedder = TimestepEmbedder(in_dim, hidden_dim)
        self.guidance_embedder = GuidanceEmbedder(in_dim, hidden_dim)

    def __call__(
        self,
        time_step: mx.array,
        guidance: mx.array,
    ) -> mx.array:
        """Compute time/guidance conditioning.

        Args:
            time_step: Timestep value [batch]
            guidance: Guidance value [batch]

        Returns:
            Conditioning tensor [batch, hidden_dim]
        """
        time_steps_proj = self._time_proj(time_step)
        time_steps_emb = self.timestep_embedder(time_steps_proj)

        guidance_proj = self._time_proj(guidance)
        guidance_emb = self.guidance_embedder(guidance_proj)

        conditioning = time_steps_emb + guidance_emb
        return conditioning.astype(ModelConfig.precision)

    @staticmethod
    def _time_proj(time_steps: mx.array) -> mx.array:
        """Project timesteps/guidance to sinusoidal embeddings.

        Args:
            time_steps: Input values [batch]

        Returns:
            Sinusoidal embeddings [batch, 256]
        """
        max_period = 10000
        half_dim = 128
        exponent = -math.log(max_period) * mx.arange(start=0, stop=half_dim, step=None, dtype=mx.float32)
        exponent = exponent / half_dim
        emb = mx.exp(exponent)
        emb = time_steps[:, None].astype(mx.float32) * emb[None, :]
        emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
        emb = mx.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)
        return emb
