import mlx.core as mx
import mlx.nn as nn

from mflux.models.depth_pro.residual_block import ResidualBlock


class FeatureFusionBlock2d(nn.Module):
    """Feature fusion block for depth estimation"""

    def __init__(self, num_features: int, deconv: bool = False, batch_norm: bool = False):
        super().__init__()

        self.resnet1 = self._residual_block(num_features, batch_norm)
        self.resnet2 = self._residual_block(num_features, batch_norm)

        self.use_deconv = deconv
        if deconv:
            self.deconv = nn.ConvTranspose2d(
                in_channels=num_features, out_channels=num_features, kernel_size=2, stride=2, padding=0, bias=False
            )

        self.out_conv = nn.Conv2d(
            in_channels=num_features, out_channels=num_features, kernel_size=1, stride=1, padding=0, bias=True
        )

    def __call__(self, x0: mx.array, x1: mx.array = None) -> mx.array:
        """Process and fuse input features"""
        x = x0

        if x1 is not None:
            res = self.resnet1(x1)
            x = x + res  # MLX handles addition directly

        x = self.resnet2(x)

        if self.use_deconv:
            x = self.deconv(x)

        x = self.out_conv(x)

        return x

    @staticmethod
    def _residual_block(num_features: int, batch_norm: bool):
        """Create a residual block"""

        def _create_block(dim: int, batch_norm: bool) -> list:
            layers = [
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=num_features,
                    out_channels=num_features,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not batch_norm,
                ),
            ]
            if batch_norm:
                # Use GroupNorm with num_groups=1 as equivalent to BatchNorm
                layers.append(nn.GroupNorm(num_groups=1, dims=dim))
            return layers

        # Create sequential residual path
        residual = nn.Sequential(
            *_create_block(dim=num_features, batch_norm=batch_norm),
            *_create_block(dim=num_features, batch_norm=batch_norm),
        )

        return ResidualBlock(residual)
