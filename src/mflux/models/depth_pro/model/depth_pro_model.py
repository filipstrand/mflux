import mlx.core as mx
import mlx.nn as nn

from mflux.models.depth_pro.model.decoder.multires_conv_decoder import MultiresConvDecoder
from mflux.models.depth_pro.model.encoder.depth_pro_encoder import DepthProEncoder
from mflux.models.depth_pro.model.head.fov_head import FOVHead


class DepthProModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DepthProEncoder()
        self.decoder = MultiresConvDecoder()
        self.head = FOVHead()

    def __call__(self, x0: mx.array, x1: mx.array, x2: mx.array) -> tuple[mx.array, mx.array]:
        x0_lat, x1_lat, x0_feat, x1_feat, x_global = self.encoder(x0, x1, x2)
        decoded = self.decoder(x0_lat, x1_lat, x0_feat, x1_feat, x_global)
        return self.head(decoded)
