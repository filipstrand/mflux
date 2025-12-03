import mlx.core as mx
import mlx.nn as nn

from mflux.models.depth_pro.model.decoder.feature_fusion_block_2d import FeatureFusionBlock2d
from mflux.models.depth_pro.model.depth_pro_util import DepthProUtil


class MultiresConvDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = [
            nn.Identity(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
        ]
        self.fusions = [
            FeatureFusionBlock2d(num_features=256, deconv=False),
            FeatureFusionBlock2d(num_features=256, deconv=True),
            FeatureFusionBlock2d(num_features=256, deconv=True),
            FeatureFusionBlock2d(num_features=256, deconv=True),
            FeatureFusionBlock2d(num_features=256, deconv=True),
        ]

    def __call__(
        self,
        x0_latent: mx.array,
        x1_latent: mx.array,
        x0_features: mx.array,
        x1_features: mx.array,
        x_global_features: mx.array,
    ) -> mx.array:
        # Process global features:
        features = DepthProUtil.apply_conv(x_global_features, self.convs[4])
        features = self.fusions[4](features)

        # Process remaining levels with skip connections:
        x1_skip_features = DepthProUtil.apply_conv(x1_features, self.convs[3])
        features = self.fusions[3](features, x1_skip_features)

        x0_skip_features = DepthProUtil.apply_conv(x0_features, self.convs[2])
        features = self.fusions[2](features, x0_skip_features)

        x1_skip_latents = DepthProUtil.apply_conv(x1_latent, self.convs[1])
        features = self.fusions[1](features, x1_skip_latents)

        x0_skip_latents = DepthProUtil.apply_conv(x0_latent, self.convs[0])
        features = self.fusions[0](features, x0_skip_latents)

        return features
