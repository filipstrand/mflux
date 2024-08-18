import math
from mlx import nn
import mlx.core as mx

from flux_1_schnell.config.config import Config
from flux_1_schnell.models.transformer.text_embedder import TextEmbedder
from flux_1_schnell.models.transformer.timestep_embedder import TimestepEmbedder
from flux_1_schnell.models.transformer.guidance_embedder import GuidanceEmbedder


class TimeTextEmbed(nn.Module):

    def __init__(self, with_guidance_embed: bool = False):
        super().__init__()
        self.text_embedder = TextEmbedder()
        self.with_guidance_embed = with_guidance_embed
        if self.with_guidance_embed:
            self.guidance_embedder = GuidanceEmbedder()
        self.timestep_embedder = TimestepEmbedder()

    def forward(self, time_step: mx.array, pooled_projection: mx.array, guidance: mx.array) -> mx.array:
        time_steps_proj = self._time_proj(time_step)
        time_steps_emb = self.timestep_embedder.forward(time_steps_proj)
        if self.with_guidance_embed:
            time_steps_emb += self.guidance_embedder.forward(self._time_proj(guidance))
        pooled_projections = self.text_embedder.forward(pooled_projection)
        conditioning = time_steps_emb + pooled_projections
        return conditioning.astype(Config.precision)

    @staticmethod
    def _time_proj(time_steps: mx.array) -> mx.array:
        max_period = 10000
        half_dim = 128
        exponent = -math.log(max_period) * mx.arange(start=0, stop=half_dim, step=None, dtype=mx.float32)
        exponent = exponent / half_dim
        emb = mx.exp(exponent)
        emb = time_steps[:, None].astype(mx.float32) * emb[None, :]
        emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
        emb = mx.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)
        return emb

