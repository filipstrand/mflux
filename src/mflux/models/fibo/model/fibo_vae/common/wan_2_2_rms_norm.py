import mlx.core as mx
from mlx import nn


class Wan2_2_RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-12, images: bool = True):
        super().__init__()
        self.eps = eps
        self.scale = float(dim) ** 0.5
        self.images = images
        if images:
            self.weight = mx.ones((dim, 1, 1))
        else:
            self.weight = mx.ones((dim, 1, 1, 1))

    def __call__(self, x: mx.array) -> mx.array:
        sum_sq = mx.sum(x * x, axis=1, keepdims=True)
        l2_norm = mx.sqrt(sum_sq)
        denom = mx.maximum(l2_norm, mx.array(self.eps, dtype=l2_norm.dtype))
        x_normalized = x / denom
        if x.ndim == 5 and not self.images:
            weight = self.weight.reshape(1, -1, 1, 1, 1)
        elif x.ndim == 4 and self.images:
            weight = self.weight.reshape(1, -1, 1, 1)
        else:
            if x.ndim == 5:
                weight = self.weight.reshape(1, -1, 1, 1, 1)
            elif x.ndim == 4:
                weight = self.weight.reshape(1, -1, 1, 1)
            else:
                weight = self.weight
        return x_normalized * self.scale * weight
