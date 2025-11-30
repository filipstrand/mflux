import mlx.core as mx
import mlx.nn as nn

from mflux.models.depth_pro.model.depth_pro_util import DepthProUtil


class ResidualBlock(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.residual = [
            nn.Identity(),
            nn.Conv2d(
                in_channels=num_features,
                out_channels=num_features,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.Identity(),
            nn.Conv2d(
                in_channels=num_features,
                out_channels=num_features,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        delta_x = nn.relu(x)
        delta_x = DepthProUtil.apply_conv(delta_x, self.residual[1])
        delta_x = nn.relu(delta_x)
        delta_x = DepthProUtil.apply_conv(delta_x, self.residual[3])
        return x + delta_x
