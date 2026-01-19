import mlx.core as mx
from mlx import nn


class QwenImageRMSNorm(nn.Module):
    """Channel-wise L2 normalization for VAE encoder/decoder.

    NOTE: Despite the name, this is NOT RMS normalization. It uses L2 norm:
        output = (x / ||x||_2) * scale * weight
    where ||x||_2 = sqrt(sum(x^2)) computed across the channel dimension (axis=1).

    This differs from standard RMSNorm which uses:
        output = x / sqrt(mean(x^2)) * weight

    Cannot use mx.fast.rms_norm() because:
    1. Different formula (L2 norm vs RMS)
    2. Normalizes across channel axis (axis=1), not last axis
    3. Has custom scaling factor (sqrt(num_channels))
    4. Dynamic weight reshaping based on input dimensions (4D vs 5D)
    """

    def __init__(self, num_channels: int, eps: float = 1e-12, images: bool = True):
        super().__init__()
        self.eps = eps
        self.scale = float(num_channels) ** 0.5
        self.images = images
        if images:
            self.weight = mx.ones((num_channels, 1, 1))
        else:
            self.weight = mx.ones((num_channels, 1, 1, 1))

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
