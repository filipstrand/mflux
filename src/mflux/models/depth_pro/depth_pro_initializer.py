from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.depth_pro.model.depth_pro_model import DepthProModel
from mflux.models.depth_pro.weights.depth_pro_weight_definition import DepthProWeightDefinition


class DepthProInitializer:
    @staticmethod
    def init(model: DepthProModel, quantize: int | None = None) -> None:
        # 1. Load weights using unified loader (handles download from Apple CDN)
        weights = WeightLoader.load(weight_definition=DepthProWeightDefinition)

        # 2. Apply weights and quantize using unified applier
        WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=DepthProWeightDefinition,
            models={
                "depth_pro": model,
            },
        )
