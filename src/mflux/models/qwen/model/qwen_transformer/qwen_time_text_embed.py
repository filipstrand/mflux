import mlx.core as mx
from mlx import nn

from mflux.models.qwen.model.qwen_transformer.qwen_timestep_embedding import QwenTimestepEmbedding
from mflux.models.qwen.model.qwen_transformer.qwen_timesteps import QwenTimesteps


class QwenTimeTextEmbed(nn.Module):
    def __init__(self, timestep_proj_dim: int = 256, inner_dim: int = 3072):
        super().__init__()
        # PyTorch: self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        self.time_proj = QwenTimesteps(proj_dim=timestep_proj_dim, scale=1000.0)

        # PyTorch: self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.timestep_embedder = QwenTimestepEmbedding(proj_dim=timestep_proj_dim, inner_dim=inner_dim)

    def __call__(self, timestep: mx.array, hidden_states: mx.array) -> mx.array:
        """
        Matches PyTorch QwenTimestepProjEmbeddings.forward exactly.

        PyTorch:
            timesteps_proj = self.time_proj(timestep)
            timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))
            conditioning = timesteps_emb
            return conditioning
        """
        # PyTorch: timesteps_proj = self.time_proj(timestep)
        timesteps_proj = self.time_proj(timestep)

        # PyTorch: timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))
        timesteps_emb = self.timestep_embedder(timesteps_proj.astype(hidden_states.dtype))

        # PyTorch: conditioning = timesteps_emb
        conditioning = timesteps_emb

        return conditioning
