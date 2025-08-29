import mlx.core as mx
from mlx import nn


class QwenImageCausalConv3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size

    def __call__(self, x: mx.array) -> mx.array:
        if isinstance(self.padding, int):
            pad_t = pad_h = pad_w = self.padding
        else:
            pad_t, pad_h, pad_w = self.padding

        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            pad_spec = [
                (0, 0),  # Batch dimension
                (0, 0),  # Channel dimension
                (2 * pad_t, 0),  # Temporal dimension (causal: 2*padding, 0)
                (pad_h, pad_h),  # Height dimension
                (pad_w, pad_w),  # Width dimension
            ]
            x = mx.pad(x, pad_spec)

        # Transpose input from PyTorch format (B, C, D, H, W) to MLX format (B, D, H, W, C) right before conv
        x = mx.transpose(x, (0, 2, 3, 4, 1))
        
        # Transpose weights from PyTorch format (out_ch, in_ch, d, h, w) to MLX format (out_ch, d, h, w, in_ch)
        original_weight = self.conv3d.weight
        if len(original_weight.shape) == 5:
            # Transpose weight for MLX Conv3d
            mlx_weight = mx.transpose(original_weight, (0, 2, 3, 4, 1))
            # Temporarily assign transposed weight
            self.conv3d.weight = mlx_weight
            x = self.conv3d(x)
            # Restore original weight
            self.conv3d.weight = original_weight
        else:
            x = self.conv3d(x)
        
        # Transpose output back from MLX format (B, D, H, W, C) to PyTorch format (B, C, D, H, W)
        x = mx.transpose(x, (0, 4, 1, 2, 3))

        return x
