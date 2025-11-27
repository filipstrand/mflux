import mlx.core as mx
from mlx import nn


class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_conv_shortcut: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = use_conv_shortcut

        self.norm1 = nn.GroupNorm(
            num_groups=32,
            dims=in_channels,
            eps=1e-6,
            affine=True,
            pytorch_compatible=True,
        )
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.norm2 = nn.GroupNorm(
            num_groups=32,
            dims=out_channels,
            eps=1e-6,
            affine=True,
            pytorch_compatible=True,
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Skip connection with 1x1 conv if channels change
        if use_conv_shortcut or in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
            )
        else:
            self.conv_shortcut = None

    def __call__(self, input_array: mx.array) -> mx.array:
        input_array = mx.transpose(input_array, (0, 2, 3, 1))
        hidden_states = self.norm1(input_array)
        hidden_states = nn.silu(hidden_states)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.norm2(hidden_states)
        hidden_states = nn.silu(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_array = self.conv_shortcut(input_array)

        output = input_array + hidden_states
        return mx.transpose(output, (0, 3, 1, 2))
