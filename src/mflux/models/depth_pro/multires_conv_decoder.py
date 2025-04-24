import mlx.core as mx
import mlx.nn as nn

from mflux.models.depth_pro.conv_utils import ConvUtils
from mflux.models.depth_pro.feature_fusion_block_2d import FeatureFusionBlock2d


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

    def __call__(self, encodings: list[mx.array]) -> tuple[mx.array, mx.array]:
        # Process last layer
        encodings_last = encodings[4]
        features = ConvUtils.apply_conv(encodings_last, self.convs[4])
        features = self.fusions[4](features)

        # Process remaining levels with skip connections
        for i in [3, 2, 1, 0]:
            enc = encodings[i]
            features_i = ConvUtils.apply_conv(enc, self.convs[i])
            features = self.fusions[i](features, features_i)

        return features
