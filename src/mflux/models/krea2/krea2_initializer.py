from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.krea2.model.krea2_text_encoder.krea2_text_encoder import Krea2TextEncoder
from mflux.models.krea2.model.krea2_transformer.transformer import Krea2Transformer
from mflux.models.krea2.model.krea2_vae.krea2_vae import Krea2VAE
from mflux.models.krea2.weights.krea2_lora_mapping import Krea2LoRAMapping
from mflux.models.krea2.weights.krea2_weight_definition import KREA2_VAE_REPO, Krea2WeightDefinition


class Krea2Initializer:
    @staticmethod
    def init(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ) -> None:
        path = model_path if model_path else model_config.model_name
        Krea2Initializer._init_config(model, model_config)
        weights = Krea2Initializer._load_weights(path)
        vae_weights = Krea2Initializer._load_vae_weights()
        Krea2Initializer._init_tokenizers(model, path)
        Krea2Initializer._init_models(model, model_config)
        Krea2Initializer._apply_weights(model, weights, vae_weights, quantize)
        Krea2Initializer._apply_lora(model, lora_paths, lora_scales)

    @staticmethod
    def _init_config(model, model_config: ModelConfig) -> None:
        model.prompt_cache = {}
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = None
        model.lora_paths = None
        model.lora_scales = None

    @staticmethod
    def _load_weights(model_path: str) -> LoadedWeights:
        return WeightLoader.load(
            weight_definition=Krea2WeightDefinition,
            model_path=model_path,
        )

    @staticmethod
    def _load_vae_weights() -> LoadedWeights:
        # Krea 2 reuses the Qwen-Image VAE, which is not bundled with the Krea checkpoint.
        vae_component = Krea2WeightDefinition.get_vae_component()
        root_path = PathResolution.resolve(path=KREA2_VAE_REPO, patterns=["vae/*.safetensors", "vae/*.json"])
        weights, _, _ = WeightLoader._load_component(root_path, vae_component)
        return LoadedWeights(components={"vae": weights}, meta_data=_empty_meta())

    @staticmethod
    def _init_tokenizers(model, model_path: str) -> None:
        model.tokenizers = TokenizerLoader.load_all(
            definitions=Krea2WeightDefinition.get_tokenizers(),
            model_path=model_path,
        )

    @staticmethod
    def _init_models(model, model_config: ModelConfig) -> None:
        model.vae = Krea2VAE()
        model.transformer = Krea2Transformer(**model_config.transformer_overrides)
        model.text_encoder = Krea2TextEncoder(**model_config.text_encoder_overrides)

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, vae_weights: LoadedWeights, quantize: int | None) -> None:
        bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=Krea2WeightDefinition,
            models={
                "transformer": model.transformer,
                "text_encoder": model.text_encoder,
            },
        )
        # VAE: load from the Qwen repo (kept in the model precision, not quantized).
        model.vae.update(vae_weights.components["vae"], strict=False)
        model.bits = bits

    @staticmethod
    def _apply_lora(
        model,
        lora_paths: list[str] | None,
        lora_scales: list[float] | None,
    ) -> None:
        # Single transformer, no CFG-unconditional: apply the LoRA to model.transformer directly.
        model.lora_paths, model.lora_scales = LoRALoader.load_and_apply_lora(
            lora_mapping=Krea2LoRAMapping.get_mapping(),
            transformer=model.transformer,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )


def _empty_meta():
    from mflux.models.common.weights.loading.loaded_weights import MetaData

    return MetaData(quantization_level=None, mflux_version=None)
