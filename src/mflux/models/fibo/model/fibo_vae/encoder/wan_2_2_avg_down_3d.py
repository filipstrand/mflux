import mlx.core as mx
from mlx import nn


class Wan2_2_AvgDown3D(nn.Module):
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

        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def __call__(self, x: mx.array) -> mx.array:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        if pad_t > 0:
            x = mx.pad(
                x,
                [
                    (0, 0),  # batch
                    (0, 0),  # channels
                    (pad_t, 0),  # time
                    (0, 0),  # height
                    (0, 0),  # width
                ],
            )

        b, c, t, h, w = x.shape

        x = mx.reshape(
            x,
            (
                b,
                c,
                t // self.factor_t,
                self.factor_t,
                h // self.factor_s,
                self.factor_s,
                w // self.factor_s,
                self.factor_s,
            ),
        )
        x = mx.transpose(x, (0, 1, 3, 5, 7, 2, 4, 6))
        x = mx.reshape(
            x,
            (
                b,
                c * self.factor,
                t // self.factor_t,
                h // self.factor_s,
                w // self.factor_s,
            ),
        )
        x = mx.reshape(
            x,
            (
                b,
                self.out_channels,
                self.group_size,
                t // self.factor_t,
                h // self.factor_s,
                w // self.factor_s,
            ),
        )
        x = mx.mean(x, axis=2)
        return x
