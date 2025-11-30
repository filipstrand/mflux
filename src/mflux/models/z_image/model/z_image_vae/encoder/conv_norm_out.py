import mlx.core as mx
from mlx import nn


class ConvNormOut(nn.Module):
    def __init__(self, num_channels: int = 512, num_groups: int = 32):
        super().__init__()
        self.norm = nn.GroupNorm(
            num_groups=num_groups, dims=num_channels, eps=1e-6, affine=True, pytorch_compatible=True
        )

    def __call__(self, input_array: mx.array) -> mx.array:
        input_array = mx.transpose(input_array, (0, 2, 3, 1))
        output = self.norm(input_array)
        return mx.transpose(output, (0, 3, 1, 2))
