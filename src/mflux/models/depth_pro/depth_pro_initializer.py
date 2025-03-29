from mflux.models.depth_pro.depth_pro import DepthPro
from mflux.models.depth_pro.weight_handler_depth_pro import WeightHandlerDepthPro


class DepthProInitializer:
    @staticmethod
    def init() -> DepthPro:
        # 1. Initialize the model
        depth_pro = DepthPro()

        # 2. Load the weights
        depth_pro_weights = WeightHandlerDepthPro.load_weights()
        WeightHandlerDepthPro.modify_encoder_weights(depth_pro_weights, "upsample_latent0")
        WeightHandlerDepthPro.modify_encoder_weights(depth_pro_weights, "upsample_latent1")
        WeightHandlerDepthPro.modify_encoder_weights(depth_pro_weights, "upsample0")
        WeightHandlerDepthPro.modify_encoder_weights(depth_pro_weights, "upsample1")
        WeightHandlerDepthPro.modify_encoder_weights(depth_pro_weights, "upsample2")

        # 3. Assign the weights to the model
        # depth_pro.depth_pro_model.update(depth_pro_weights.weights)

        return depth_pro
