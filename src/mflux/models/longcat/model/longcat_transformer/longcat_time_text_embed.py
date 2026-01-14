"""
Time and text embedding for LongCat model.

LongCat uses Qwen2.5-VL pooled embeddings (3584 dim) and does not use guidance embeddings.
"""

import math

import mlx.core as mx
from mlx import nn

from mflux.models.common.config import ModelConfig
from mflux.models.longcat.model.longcat_transformer.longcat_text_embedder import LongCatTextEmbedder
from mflux.models.flux.model.flux_transformer.timestep_embedder import TimestepEmbedder


class LongCatTimeTextEmbed(nn.Module):
    """
    Time and text embedding module for LongCat.

    Combines timestep embeddings with pooled text embeddings from Qwen2.5-VL.
    LongCat does not use guidance embeddings (guidance_embeds: false).
    """

    def __init__(self):
        super().__init__()
        self.text_embedder = LongCatTextEmbedder()
        self.timestep_embedder = TimestepEmbedder()
        # No guidance_embedder - LongCat has guidance_embeds: false

    def __call__(
        self,
        time_step: mx.array,
        pooled_projection: mx.array,
    ) -> mx.array:
        """
        Compute combined time and text embeddings.

        Args:
            time_step: Timestep values
            pooled_projection: Pooled text embeddings from Qwen2.5-VL (3584 dim)

        Returns:
            Combined conditioning embeddings (3072 dim)
        """
        time_steps_proj = self._time_proj(time_step)
        time_steps_emb = self.timestep_embedder(time_steps_proj)
        # No guidance embedding for LongCat
        pooled_projections = self.text_embedder(pooled_projection)
        conditioning = time_steps_emb + pooled_projections
        return conditioning.astype(ModelConfig.precision)

    @staticmethod
    def _time_proj(time_steps: mx.array) -> mx.array:
        """Convert timesteps to sinusoidal embeddings."""
        max_period = 10000
        half_dim = 128
        exponent = -math.log(max_period) * mx.arange(start=0, stop=half_dim, step=None, dtype=mx.float32)
        exponent = exponent / half_dim
        emb = mx.exp(exponent)
        emb = time_steps[:, None].astype(mx.float32) * emb[None, :]
        emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
        emb = mx.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)
        return emb
