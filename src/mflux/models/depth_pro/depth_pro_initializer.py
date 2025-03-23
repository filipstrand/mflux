import mlx.nn as nn

from mflux.models.depth_pro.depth_pro import DepthPro
from mflux.models.depth_pro.weight_handler_depth_pro import WeightHandlerDepthPro


class DepthProInitializer:
    @staticmethod
    def init(quantize: int | None = None) -> DepthPro:
        # 1. Initialize the model
        depth_pro = DepthPro()

        # 2. Load the weights
        depth_pro_weights = WeightHandlerDepthPro.load_weights()
        WeightHandlerDepthPro.reposition_encoder_weights(depth_pro_weights, "upsample_latent0")
        WeightHandlerDepthPro.reposition_encoder_weights(depth_pro_weights, "upsample_latent1")
        WeightHandlerDepthPro.reposition_encoder_weights(depth_pro_weights, "upsample0")
        WeightHandlerDepthPro.reposition_encoder_weights(depth_pro_weights, "upsample1")
        WeightHandlerDepthPro.reposition_encoder_weights(depth_pro_weights, "upsample2")
        WeightHandlerDepthPro.reposition_head_weights(depth_pro_weights)
        WeightHandlerDepthPro.reshape_transposed_convolution_weights(depth_pro_weights)

        # 3. Assign the weights to the model
        depth_pro.depth_pro_model.update(depth_pro_weights.weights)

        # 4. Optionally quantize the model
        if quantize:
            nn.quantize(depth_pro.depth_pro_model, bits=quantize)

        return depth_pro
