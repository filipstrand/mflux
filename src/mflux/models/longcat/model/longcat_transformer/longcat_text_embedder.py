"""
Text embedder for LongCat model.

Converts Qwen2.5-VL pooled text embeddings (3584 dim) to transformer hidden dimension (3072).
"""

import mlx.core as mx
from mlx import nn


class LongCatTextEmbedder(nn.Module):
    """
    Text embedder for LongCat.

    Takes pooled text embeddings from Qwen2.5-VL (3584 dim) and projects
    to the transformer hidden dimension (3072).
    """

    def __init__(self):
        super().__init__()
        # Qwen2.5-VL hidden_size (3584) -> transformer hidden (3072)
        self.linear_1 = nn.Linear(3584, 3072)
        self.linear_2 = nn.Linear(3072, 3072)

    def __call__(self, caption: mx.array) -> mx.array:
        hidden_states = self.linear_1(caption)
        hidden_states = nn.silu(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states
