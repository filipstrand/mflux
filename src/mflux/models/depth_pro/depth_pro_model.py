import mlx.core as mx
import mlx.nn as nn

from mflux.models.depth_pro.depth_pro_encoder import DepthProEncoder
from mflux.models.depth_pro.dino_v2.dino_vision_transformer import DinoVisionTransformer
from mflux.models.depth_pro.fov_head import FOVHead
from mflux.models.depth_pro.multires_conv_decoder import MultiresConvDecoder


class DepthProModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DepthProEncoder()
        self.decoder = MultiresConvDecoder()
        self.head = FOVHead()
        self.fov = DinoVisionTransformer()

    def __call__(self, x: mx.array) -> (mx.array, mx.array):
        encodings = self.encoder(x)
        features, low_res_features = self.decoder(encodings)
        canonical_inverse_depth = self.head(features)
        fov_deg = self.fov(x, low_res_features)
        return canonical_inverse_depth, fov_deg
