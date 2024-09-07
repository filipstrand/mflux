import mlx.core as mx
from mlx import nn

from mflux.config.config import Config


class ResnetBlock2D(nn.Module):

    def __init__(
            self,
            norm1: int,
            conv1_in: int,
            conv1_out: int,
            norm2: int,
            conv2_in: int,
            conv2_out: int,
            conv_shortcut_in: int | None = None,
            conv_shortcut_out: int | None = None,
            is_conv_shortcut: bool = False
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(
            num_groups=32,
            dims=norm1,
            eps=1e-6,
            affine=True,
            pytorch_compatible=True
        )
        self.norm2 = nn.GroupNorm(
            num_groups=32,
            dims=norm2,
            eps=1e-6,
            affine=True,
            pytorch_compatible=True
        )
        self.conv1 = nn.Conv2d(
            in_channels=conv1_in,
            out_channels=conv1_out,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.conv2 = nn.Conv2d(
            in_channels=conv2_in,
            out_channels=conv2_out,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.is_conv_shortcut = is_conv_shortcut
        self.conv_shortcut = None if not is_conv_shortcut else nn.Conv2d(
            in_channels=conv_shortcut_in,
            out_channels=conv_shortcut_out,
            kernel_size=(1, 1),
            stride=(1, 1),
        )

    def forward(self, input_array: mx.array) -> mx.array:
        input_array = mx.transpose(input_array, (0, 2, 3, 1))
        hidden_states = self.norm1(input_array.astype(mx.float32)).astype(Config.precision)
        hidden_states = nn.silu(hidden_states)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.norm2(hidden_states.astype(mx.float32)).astype(Config.precision)
        hidden_states = nn.silu(hidden_states)
        hidden_states = self.conv2(hidden_states)
        if self.is_conv_shortcut:
            input_array = self.conv_shortcut(input_array)
        output_tensor = input_array + hidden_states
        return mx.transpose(output_tensor, (0, 3, 1, 2))
