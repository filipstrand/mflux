import mlx.core as mx
import mlx.nn as nn

from mflux.models.depth_pro.model.decoder.residual_block import ResidualBlock
from mflux.models.depth_pro.model.depth_pro_util import DepthProUtil


class FeatureFusionBlock2d(nn.Module):
    def __init__(self, num_features: int, deconv: bool = False):
        super().__init__()
        self.use_deconv = deconv
        self.resnet1 = ResidualBlock(num_features)
        self.resnet2 = ResidualBlock(num_features)
        self.deconv = nn.ConvTranspose2d(in_channels=num_features, out_channels=num_features, kernel_size=2, stride=2, padding=0, bias=False)  # fmt: off
        self.out_conv = nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=1, stride=1, padding=0, bias=True)  # fmt: off

    def __call__(self, x0: mx.array, x1: mx.array | None = None) -> mx.array:
        x = x0
        if x1 is not None:
            res = self.resnet1(x1)
            x = x + res
        x = self.resnet2(x)
        if self.use_deconv:
            x = DepthProUtil.apply_conv(x, self.deconv)
        x = DepthProUtil.apply_conv(x, self.out_conv)
        return x
