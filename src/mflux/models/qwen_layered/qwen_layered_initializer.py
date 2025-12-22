import mlx.core as mx

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.qwen.model.qwen_text_encoder.qwen_text_encoder import QwenTextEncoder
from mflux.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer  # Use base transformer
from mflux.models.qwen_layered.model.qwen_layered_vae.qwen_layered_vae import QwenLayeredVAE
from mflux.models.qwen_layered.weights.qwen_layered_weight_definition import QwenLayeredWeightDefinition


class QwenLayeredInitializer:
    """Initializer for Qwen-Image-Layered model."""

    @staticmethod
    def init(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ) -> None:
        """Initialize the Qwen-Image-Layered model."""
        path = model_path if model_path else model_config.model_name

        QwenLayeredInitializer._init_config(model, model_config)
        weights = QwenLayeredInitializer._load_weights(path)
        QwenLayeredInitializer._init_tokenizers(model, path)
        QwenLayeredInitializer._init_models(model)
        QwenLayeredInitializer._apply_weights(model, weights, quantize)
        # Note: LoRA not supported yet for layered model

    @staticmethod
    def _init_config(model, model_config: ModelConfig) -> None:
        model.model_config = model_config
        model.lora_paths = None
        model.lora_scales = None
        model.prompt_cache = {}
        model.callbacks = CallbackRegistry()
        model.bits = None

    @staticmethod
    def _load_weights(path: str) -> LoadedWeights:
        return WeightLoader.load(
            weight_definition=QwenLayeredWeightDefinition,
            model_path=path,
        )

    @staticmethod
    def _init_tokenizers(model, path: str) -> None:
        model.tokenizers = TokenizerLoader.load_all(
            definitions=QwenLayeredWeightDefinition.get_tokenizers(),
            model_path=path,
        )

    @staticmethod
    def _init_models(model) -> None:
        """Initialize model components."""
        model.vae = QwenLayeredVAE(input_channels=4, output_channels=4)
        model.transformer = QwenTransformer()  # Use base transformer
        model.text_encoder = QwenTextEncoder()

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None) -> None:
        """Apply weights and optionally quantize."""
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=QwenLayeredWeightDefinition,
            models={
                "vae": model.vae,
                "transformer": model.transformer,
                "text_encoder": model.text_encoder,
            },
        )

        # Evaluate to load weights into memory
        mx.eval(model.parameters())
