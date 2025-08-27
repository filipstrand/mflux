import mlx.core as mx
from mlx import nn

from mflux.models.qwen.model.qwen_transformer.qwen_timestep_embedding import QwenTimestepEmbedding
from mflux.models.qwen.model.qwen_transformer.qwen_timesteps import QwenTimesteps


class QwenTimeTextEmbed(nn.Module):
    def __init__(self, timestep_proj_dim: int = 256, inner_dim: int = 3072):
        super().__init__()
        self.time_proj = QwenTimesteps(proj_dim=timestep_proj_dim)
        self.timestep_embedder = QwenTimestepEmbedding(proj_dim=timestep_proj_dim, inner_dim=inner_dim)

    def __call__(self, timestep: mx.array, hidden_states: mx.array) -> mx.array:
        time_proj = self.time_proj(timestep)
        time_emb = self.timestep_embedder(time_proj.astype(hidden_states.dtype))
        return time_emb
