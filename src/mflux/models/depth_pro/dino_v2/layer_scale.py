import mlx.core as mx
import mlx.nn as nn


class LayerScale(nn.Module):
    def __init__(
        self,
        dims: int,
        init_values: float = 1e-5,
    ) -> None:
        super().__init__()
        self.gamma = init_values * mx.ones((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.gamma
