"""Qwen3VL Vision Patch Merger for MLX."""

import mlx.core as mx
import numpy as np
from mlx import nn


class Qwen3VLVisionPatchMerger(nn.Module):
    """Patch merger for Qwen3VL vision transformer."""

    def __init__(
        self,
        hidden_size: int = 1024,
        spatial_merge_size: int = 2,
        out_hidden_size: int = 2560,
        use_postshuffle_norm: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size * (spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(self.hidden_size if use_postshuffle_norm else hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = lambda x: 0.5 * x * (1 + mx.tanh(mx.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))  # GELU
        self.linear_fc2 = nn.Linear(self.hidden_size, out_hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Matches PyTorch behavior:
        - Input: (seq_len, hidden_size) where hidden_size=1024
        - Normalize over hidden_size (1024)
        - Reshape to (-1, merged_hidden_size) where merged_hidden_size=4096
        - Process through linear layers
        """
        if self.use_postshuffle_norm:
            x = self.norm(x.reshape(-1, self.hidden_size)).reshape(-1, self.hidden_size)
        else:
            # Normalize over original hidden_size (1024), then reshape to merged size (4096)
            # PyTorch: self.norm(x).view(-1, self.hidden_size) where self.hidden_size=4096
            x = self.norm(x)  # Normalize over last dim (1024)
            # Reshape to (-1, merged_hidden_size) = (-1, 4096)
            # This flattens and groups: (836, 1024) -> (209, 4096)
            x = x.reshape(-1, self.hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x
