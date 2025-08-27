import mlx.core as mx
import mlx.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(in_channels=3, out_channels=1024, kernel_size=16, stride=16, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.transpose(x, (0, 2, 3, 1))
        x = self.proj(x)
        return x
