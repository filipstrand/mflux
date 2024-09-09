import math
from mlx import nn
import mlx.core as mx

from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.models.transformer.text_embedder import TextEmbedder
from mflux.models.transformer.timestep_embedder import TimestepEmbedder
from mflux.models.transformer.guidance_embedder import GuidanceEmbedder


class TimeTextEmbed(nn.Module):

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.text_embedder = TextEmbedder()
        self.guidance_embedder = GuidanceEmbedder() if model_config == ModelConfig.FLUX1_DEV else None
        self.timestep_embedder = TimestepEmbedder()

    def forward(self, time_step: mx.array, pooled_projection: mx.array, guidance: mx.array) -> mx.array:
        time_steps_proj = self._time_proj(time_step)
        time_steps_emb = self.timestep_embedder.forward(time_steps_proj)
        if self.guidance_embedder is not None:
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

