import mlx.core as mx
import mlx.nn as nn


class DINOv3LayerScale(nn.Module):
    """Learnable per-channel scaling, same pattern as DINOv2."""

    def __init__(self, dims: int = 4096, init_values: float = 1.0):
        super().__init__()
        self.gamma = init_values * mx.ones((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.gamma
