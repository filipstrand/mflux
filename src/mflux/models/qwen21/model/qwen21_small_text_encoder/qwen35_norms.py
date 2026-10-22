import mlx.core as mx
from mlx import nn


class Qwen35RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        # Zero-centred: the checkpoint stores w and the norm scales by (1 + w).
        self.weight = mx.zeros((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        scale = 1.0 + self.weight.astype(mx.float32)
        return mx.fast.rms_norm(x.astype(mx.float32), scale, self.eps).astype(x.dtype)


class Qwen35RMSNormGated(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        normed = mx.fast.rms_norm(x.astype(mx.float32), None, self.eps).astype(x.dtype)
        return ((self.weight * normed) * nn.silu(gate.astype(mx.float32))).astype(x.dtype)
