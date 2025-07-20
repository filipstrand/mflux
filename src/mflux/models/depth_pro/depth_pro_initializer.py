import mlx.nn as nn

from mflux.models.depth_pro.depth_pro_model import DepthProModel
from mflux.models.depth_pro.weight_handler_depth_pro import WeightHandlerDepthPro


class DepthProInitializer:
    @staticmethod
    def init(depth_pro_model: DepthProModel, quantize: int | None = None) -> None:
        # 1. Load the weights
        depth_pro_weights = WeightHandlerDepthPro.load_weights()
        WeightHandlerDepthPro.reposition_encoder_weights(depth_pro_weights, "upsample_latent0")
        WeightHandlerDepthPro.reposition_encoder_weights(depth_pro_weights, "upsample_latent1")
        WeightHandlerDepthPro.reposition_encoder_weights(depth_pro_weights, "upsample0")
        WeightHandlerDepthPro.reposition_encoder_weights(depth_pro_weights, "upsample1")
        WeightHandlerDepthPro.reposition_encoder_weights(depth_pro_weights, "upsample2")
        WeightHandlerDepthPro.reposition_head_weights(depth_pro_weights)
        WeightHandlerDepthPro.reshape_transposed_convolution_weights(depth_pro_weights)

        # 2. Assign the weights to the model
        depth_pro_model.update(depth_pro_weights.weights)

        # 3. Optionally quantize the model
        if quantize:
            nn.quantize(depth_pro_model, bits=quantize)
