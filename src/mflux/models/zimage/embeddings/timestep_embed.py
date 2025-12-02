import math

import mlx.core as mx
import mlx.nn as nn


class TimestepEmbed(nn.Module):
    """Sinusoidal timestep embedding with MLP projection.

    Converts scalar timestep to [B, embed_dim] embedding for AdaLN conditioning.
    HF structure: sinusoidal(256) -> Linear(256, 1024) -> SiLU -> Linear(1024, 256)
    """

    EMBED_DIM = 256  # Timestep embedding dimension (from HF weights)
    MLP_HIDDEN = 1024  # MLP hidden dimension
    T_SCALE = 1000.0  # From Z-Image config

    def __init__(self):
        super().__init__()
        # MLP: Linear -> SiLU -> Linear
        # Maps to t_embedder.mlp.{0,2}.weight
        self.mlp = nn.Sequential(
            nn.Linear(self.EMBED_DIM, self.MLP_HIDDEN),
            nn.SiLU(),
            nn.Linear(self.MLP_HIDDEN, self.EMBED_DIM),
        )

    def __call__(self, t: mx.array) -> mx.array:
        """Generate timestep embedding.

        Args:
            t: Timestep values [B] in [0, 1]

        Returns:
            Timestep embeddings [B, embed_dim=256]
        """
        # Scale timestep
        t = t * self.T_SCALE

        # Sinusoidal embedding
        half_dim = self.EMBED_DIM // 2
        freqs = mx.exp(-math.log(10000.0) * mx.arange(0, half_dim) / half_dim)

        # Ensure t has correct shape for broadcasting
        if t.ndim == 0:
            t = t.reshape(1)
        if t.ndim == 1:
            t = t[:, None]  # [B, 1]

        args = t * freqs[None, :]  # [B, half_dim]
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)  # [B, embed_dim]

        # MLP projection
        return self.mlp(embedding)
