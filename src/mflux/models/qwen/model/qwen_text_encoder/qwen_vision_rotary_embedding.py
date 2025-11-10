import mlx.core as mx
from mlx import nn


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        # Create inverse frequencies
        inv_freq = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        self.inv_freq = inv_freq

    def __call__(self, max_grid_size: int) -> mx.array:
        # Create position embeddings for max_grid_size
        # Returns only freqs (NOT cos/sin concatenated) - matching HF
        positions = mx.arange(max_grid_size, dtype=mx.float32)
        freqs = mx.outer(positions, self.inv_freq)
        return freqs  # [max_grid_size, dim//2]
