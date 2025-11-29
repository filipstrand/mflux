import mlx.core as mx
import mlx.nn as nn


class ConvUtils:
    @staticmethod
    def apply_conv(x: mx.array, conv_module: nn.Module) -> mx.array:
        x = mx.transpose(x, (0, 2, 3, 1))
        x = conv_module(x)
        return mx.transpose(x, (0, 3, 1, 2))
