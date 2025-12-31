from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.seedvr2.model.seedvr2_transformer.transformer import SeedVR2Transformer
from mflux.models.seedvr2.model.seedvr2_vae.vae import SeedVR2VAE
from mflux.models.seedvr2.weights.seedvr2_weight_definition import SeedVR2WeightDefinition


class SeedVR2Initializer:
    @staticmethod
    def init(
        model,
        model_config: ModelConfig,
        quantize: int | None = None,
        model_path: str | None = None,
    ) -> None:
        path = model_path if model_path else model_config.model_name
        SeedVR2Initializer._init_config(model, model_config)
        weights = SeedVR2Initializer._load_weights(path)
        SeedVR2Initializer._init_models(model)
        SeedVR2Initializer._apply_weights(model, weights, quantize)

    @staticmethod
    def _init_config(model, model_config: ModelConfig) -> None:
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = TilingConfig()

    @staticmethod
    def _load_weights(model_path: str) -> LoadedWeights:
        return WeightLoader.load(
            weight_definition=SeedVR2WeightDefinition,
            model_path=model_path,
        )

    @staticmethod
    def _init_models(model) -> None:
        model.vae = SeedVR2VAE()
        model.transformer = SeedVR2Transformer()

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None) -> None:
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=SeedVR2WeightDefinition,
            models={
                "transformer": model.transformer,
                "vae": model.vae,
            },
        )
