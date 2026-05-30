import mlx.core as mx
from mlx.utils import tree_unflatten

from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.z_image.model.z_image_text_encoder.text_encoder import TextEncoder
from mflux.models.z_image.model.z_image_transformer.control_transformer import ZImageControlTransformer
from mflux.models.z_image.model.z_image_vae.vae import VAE
from mflux.models.z_image.weights.z_image_lora_mapping import ZImageLoRAMapping
from mflux.models.z_image.weights.z_image_weight_definition import ZImageWeightDefinition


class ZImageControlInitializer:
    """Initializer for the Z-Image strict pose ControlNet variant (sc-2257).

    Identical to :class:`ZImageInitializer` except the transformer is a
    :class:`ZImageControlTransformer` and, after the base Z-Image-Turbo weights are
    applied (non-strict, so the extra control params are tolerated), the
    Fun-Controlnet-Union safetensors is overlaid onto the control branch.
    """

    @staticmethod
    def init(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        control_weights_path: str,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ) -> None:
        path = model_path if model_path else model_config.model_name
        ZImageControlInitializer._init_config(model, model_config)
        weights = ZImageControlInitializer._load_weights(path)
        ZImageControlInitializer._init_tokenizers(model, path)
        ZImageControlInitializer._init_models(model)
        ZImageControlInitializer._apply_weights(model, weights, quantize)
        ZImageControlInitializer._apply_control_weights(model, control_weights_path)
        ZImageControlInitializer._apply_lora(model, lora_paths, lora_scales)

    @staticmethod
    def _init_config(model, model_config: ModelConfig) -> None:
        from mflux.callbacks.callback_registry import CallbackRegistry

        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = None

    @staticmethod
    def _load_weights(model_path: str) -> LoadedWeights:
        return WeightLoader.load(weight_definition=ZImageWeightDefinition, model_path=model_path)

    @staticmethod
    def _init_tokenizers(model, model_path: str) -> None:
        model.tokenizers = TokenizerLoader.load_all(
            definitions=ZImageWeightDefinition.get_tokenizers(),
            model_path=model_path,
        )

    @staticmethod
    def _init_models(model) -> None:
        model.vae = VAE()
        model.transformer = ZImageControlTransformer()
        model.text_encoder = TextEncoder()

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None) -> None:
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=ZImageWeightDefinition,
            models={
                "vae": model.vae,
                "transformer": model.transformer,
                "text_encoder": model.text_encoder,
            },
        )

    @staticmethod
    def _apply_control_weights(model, control_weights_path: str) -> None:
        # The Fun-Controlnet-Union safetensors keys (control_layers.*,
        # control_noise_refiner.*, control_all_x_embedder.*) map 1:1 onto the
        # ZImageControlTransformer param tree, so a direct tree_unflatten + update
        # is sufficient. Cast to the model precision (CN ships bf16) to match.
        raw = mx.load(str(control_weights_path))
        cast = [(k, v.astype(ModelConfig.precision)) for k, v in raw.items()]
        model.transformer.update(tree_unflatten(cast))
        mx.eval(model.transformer.parameters())

    @staticmethod
    def _apply_lora(model, lora_paths: list[str] | None, lora_scales: list[float] | None) -> None:
        model.lora_paths, model.lora_scales = LoRALoader.load_and_apply_lora(
            lora_mapping=ZImageLoRAMapping.get_mapping(),
            transformer=model.transformer,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )
