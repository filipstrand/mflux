import mlx.core as mx
from mlx import nn


class Wan2_2_DupUp3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t: int,
        factor_s: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s
        self.repeats = out_channels * self.factor // in_channels

    def __call__(self, x: mx.array, first_chunk: bool = False) -> mx.array:
        b, c, t, h, w = x.shape
        x = mx.repeat(x, self.repeats, axis=1)
        x = mx.reshape(
            x,
            (
                b,
                self.out_channels,
                self.factor_t,
                self.factor_s,
                self.factor_s,
                t,
                h,
                w,
            ),
        )

        x = mx.transpose(x, (0, 1, 5, 2, 6, 3, 7, 4))
        x = mx.reshape(
            x,
            (
                b,
                self.out_channels,
                t * self.factor_t,
                h * self.factor_s,
                w * self.factor_s,
            ),
        )

        if first_chunk and self.factor_t > 1:
            x = x[:, :, self.factor_t - 1 :, :, :]

        return x
