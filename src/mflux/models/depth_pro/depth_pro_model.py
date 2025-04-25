import mlx.core as mx
import mlx.nn as nn

from mflux.models.depth_pro.depth_pro_encoder import DepthProEncoder
from mflux.models.depth_pro.fov_head import FOVHead
from mflux.models.depth_pro.multires_conv_decoder import MultiresConvDecoder


class DepthProModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DepthProEncoder()
        self.decoder = MultiresConvDecoder()
        self.head = FOVHead()

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        encodings = self.encoder(x)
        features = self.decoder(encodings)
        return self.head(features)
