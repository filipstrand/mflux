import mlx.core as mx
from mlx import nn

from mflux.models.seedvr2.model.seedvr2_vae.common.conv3d import CausalConv3d


class Downsample3D(nn.Module):
    def __init__(
        self,
        channels: int,
        spatial_only: bool = False,
    ):
        super().__init__()
        kt, st, pt = (1, 1, 0) if spatial_only else (3, 2, 1)
        self.conv = CausalConv3d(
            channels,
            channels,
            kernel_size=(kt, 3, 3),
            stride=(st, 2, 2),
            padding=(pt, 0, 0),
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.pad(x, [(0, 0), (0, 0), (0, 0), (0, 1), (0, 1)])
        return self.conv(x)
