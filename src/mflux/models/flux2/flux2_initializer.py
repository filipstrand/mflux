from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.models.flux2.weights.flux2_lora_mapping import Flux2LoRAMapping
from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition


class Flux2Initializer:
    _KLEIN_TRANSFORMER_CONFIGS = {
        "flux2-klein-4b": {
            "patch_size": 1,
            "in_channels": 128,
            "num_layers": 5,
            "num_single_layers": 20,
            "attention_head_dim": 128,
            "num_attention_heads": 24,
            "joint_attention_dim": 7680,
            "timestep_guidance_channels": 256,
            "mlp_ratio": 3.0,
            "axes_dims_rope": (32, 32, 32, 32),
            "rope_theta": 2000,
            "guidance_embeds": False,
        },
        "flux2-klein-9b": {
            "patch_size": 1,
            "in_channels": 128,
            "num_layers": 8,
            "num_single_layers": 24,
            "attention_head_dim": 128,
            "num_attention_heads": 32,
            "joint_attention_dim": 12288,
            "timestep_guidance_channels": 256,
            "mlp_ratio": 3.0,
            "axes_dims_rope": (32, 32, 32, 32),
            "rope_theta": 2000,
            "guidance_embeds": False,
        },
    }

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
        Flux2Initializer._init_config(model, model_config)
        weights = Flux2Initializer._load_weights(path)
        Flux2Initializer._init_tokenizers(model, path)
        Flux2Initializer._init_models(model)
        Flux2Initializer._apply_weights(model, weights, quantize)
        Flux2Initializer._apply_lora(model, lora_paths, lora_scales)

    @staticmethod
    def _init_config(model, model_config: ModelConfig) -> None:
        model.prompt_cache = {}
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = None

    @staticmethod
    def _load_weights(model_path: str) -> LoadedWeights:
        return WeightLoader.load(
            weight_definition=Flux2KleinWeightDefinition,
            model_path=model_path,
        )

    @staticmethod
    def _init_tokenizers(model, model_path: str) -> None:
        model.tokenizers = TokenizerLoader.load_all(
            definitions=Flux2KleinWeightDefinition.get_tokenizers(),
            model_path=model_path,
        )

    @staticmethod
    def _init_models(model) -> None:
        model.vae = Flux2VAE()
        transformer_config = Flux2Initializer._get_transformer_config(model.model_config)
        model.transformer = Flux2Transformer(**transformer_config)
        model.text_encoder = Qwen3TextEncoder()

    @staticmethod
    def _get_transformer_config(model_config: ModelConfig) -> dict:
        model_id = model_config.model_name.lower()
        if "klein-9b" in model_id or "klein-base-9b" in model_id:
            return Flux2Initializer._KLEIN_TRANSFORMER_CONFIGS["flux2-klein-9b"]
        return Flux2Initializer._KLEIN_TRANSFORMER_CONFIGS["flux2-klein-4b"]

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None) -> None:
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=Flux2KleinWeightDefinition,
            models={
                "vae": model.vae,
                "transformer": model.transformer,
                "text_encoder": model.text_encoder,
            },
        )

    @staticmethod
    def _apply_lora(model, lora_paths: list[str] | None, lora_scales: list[float] | None) -> None:
        model.lora_paths, model.lora_scales = LoRALoader.load_and_apply_lora(
            lora_mapping=Flux2LoRAMapping.get_mapping(),
            transformer=model.transformer,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )
