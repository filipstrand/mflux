import mlx.core as mx
import mlx.nn as nn


class ConvUtils:
    @staticmethod
    def apply_conv(x: mx.array, conv_module: nn.Module) -> mx.array:
        """Apply a convolution with channel format conversion.

        MLX expects channels-last format (B,H,W,C) for convolutions,
        but tensors are generally in channels-first format (B,C,H,W).
        This helper handles the conversion automatically.

        Args:
            x: Input tensor in channels-first format (B,C,H,W)
            conv_module: Convolution module to apply

        Returns:
            Output tensor in channels-first format (B,C,H,W)
        """
        x = mx.transpose(x, (0, 2, 3, 1))
        x = conv_module(x)
        return mx.transpose(x, (0, 3, 1, 2))
