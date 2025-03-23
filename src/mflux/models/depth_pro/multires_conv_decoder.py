import mlx.core as mx
import mlx.nn as nn

from mflux.models.depth_pro.feature_fusion_block_2d import FeatureFusionBlock2d


class MultiresConvDecoder(nn.Module):
    """Decoder for multi-resolution encodings from DepthProEncoder"""

    def __init__(self):
        super().__init__()
        # Default values, these should be the same as in the encoder
        self.dims_encoder = [256, 512, 1024, 1024]
        self.dim_decoder = 256
        self.dim_out = self.dim_decoder

        num_encoders = len(self.dims_encoder)

        # Create projection convolutions
        # For highest resolution (level 0), use 1x1 conv if dimensions don't match,
        # otherwise use identity
        if self.dims_encoder[0] != self.dim_decoder:
            conv0 = nn.Conv2d(
                in_channels=self.dims_encoder[0],
                out_channels=self.dim_decoder,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
        else:
            conv0 = nn.Identity()

        convs = [conv0]

        # For other levels, use 3x3 convs
        for i in range(1, num_encoders):
            convs.append(
                nn.Conv2d(
                    in_channels=self.dims_encoder[i],
                    out_channels=self.dim_decoder,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
            )

        self.convs = convs

        # Create fusion blocks
        fusions = []
        for i in range(num_encoders):
            fusions.append(
                FeatureFusionBlock2d(
                    num_features=self.dim_decoder,
                    deconv=(i != 0),  # Use deconv for all except highest resolution
                    batch_norm=False,
                )
            )

        self.fusions = fusions

    def __call__(self, encodings: list[mx.array]) -> tuple[mx.array, mx.array]:
        """Decode the multi-resolution encodings"""
        num_levels = len(encodings)
        num_encoders = len(self.dims_encoder)

        if num_levels != num_encoders:
            raise ValueError(f"Got encoder output levels={num_levels}, expected levels={num_encoders}.")

        # Project and fuse features from lowest resolution to highest
        # Start with lowest resolution features
        features = self.convs[-1](encodings[-1])
        lowres_features = features  # Save for FOV estimation

        # Apply first fusion block (without skip connection)
        features = self.fusions[-1](features)

        # Process remaining levels with skip connections
        for i in range(num_levels - 2, -1, -1):
            features_i = self.convs[i](encodings[i])
            features = self.fusions[i](features, features_i)

        return features, lowres_features
