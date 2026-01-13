"""FLUX.2 model initializer.

Handles loading and initialization of FLUX.2 components:
- Mistral3 text encoder (single encoder)
- 32-channel VAE
- FLUX.2 transformer (8 joint + 48 single blocks)
- LoRA support
"""

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.flux2.model.flux2_text_encoder import Mistral3TextEncoder
from mflux.models.flux2.model.flux2_transformer import Flux2Transformer
from mflux.models.flux2.model.flux2_vae import Flux2VAE
from mflux.models.flux2.weights import Flux2LoRAMapping, Flux2WeightDefinition


class Flux2Initializer:
    """Initializer for FLUX.2 model.

    Handles loading weights, tokenizers, and applying quantization/LoRA
    for the FLUX.2 32B parameter model.
    """

    @staticmethod
    def init(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        custom_transformer=None,
    ) -> None:
        """Initialize a FLUX.2 model with all components.

        Args:
            model: The model instance to initialize
            model_config: Model configuration
            quantize: Quantization bits (4 or 8) or None for full precision
            model_path: Optional custom path to model weights
            lora_paths: Optional list of LoRA adapter paths
            lora_scales: Optional list of LoRA scales (one per path)
            custom_transformer: Optional custom transformer implementation
        """
        path = model_path if model_path else model_config.model_name
        Flux2Initializer._init_config(model, model_config)
        weights = Flux2Initializer._load_weights(path)
        Flux2Initializer._init_tokenizers(model, path, model_config)
        Flux2Initializer._init_models(model, model_config, weights, custom_transformer)
        Flux2Initializer._apply_weights(model, weights, quantize)
        Flux2Initializer._apply_lora(model, lora_paths, lora_scales)

    @staticmethod
    def _init_config(model, model_config: ModelConfig) -> None:
        """Initialize model configuration and callbacks.

        Args:
            model: Model instance to configure
            model_config: Model configuration
        """
        model.prompt_cache = {}
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = None

    @staticmethod
    def _load_weights(model_path: str) -> LoadedWeights:
        """Load model weights from disk or HuggingFace Hub.

        Args:
            model_path: Path or HuggingFace repo ID

        Returns:
            LoadedWeights containing all component weights
        """
        return WeightLoader.load(
            weight_definition=Flux2WeightDefinition,
            model_path=model_path,
        )

    @staticmethod
    def _init_tokenizers(model, model_path: str, model_config: ModelConfig) -> None:
        """Initialize tokenizers from model path.

        FLUX.2 uses Mistral3 tokenizer (PreTrainedTokenizerFast).

        Args:
            model: Model instance
            model_path: Path or HuggingFace repo ID
            model_config: Model configuration with optional max_sequence_length
        """
        max_length_overrides = (
            {"mistral3": model_config.max_sequence_length}
            if model_config.max_sequence_length is not None
            else {}
        )
        model.tokenizers = TokenizerLoader.load_all(
            definitions=Flux2WeightDefinition.get_tokenizers(),
            model_path=model_path,
            max_length_overrides=max_length_overrides,
        )

    @staticmethod
    def _init_models(
        model,
        model_config: ModelConfig,
        weights: LoadedWeights,
        custom_transformer=None,
    ) -> None:
        """Initialize model components (VAE, encoder, transformer).

        Args:
            model: Model instance
            model_config: Model configuration
            weights: Loaded weights to determine block counts
            custom_transformer: Optional custom transformer implementation
        """
        model.vae = Flux2VAE()
        model.text_encoder = Mistral3TextEncoder()

        if custom_transformer is not None:
            model.transformer = custom_transformer
        else:
            model.transformer = Flux2Transformer(
                model_config=model_config,
                num_transformer_blocks=weights.num_transformer_blocks(),
                num_single_transformer_blocks=weights.num_single_transformer_blocks(),
            )

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None) -> None:
        """Apply loaded weights to model components with optional quantization.

        Args:
            model: Model instance with components
            weights: LoadedWeights to apply
            quantize: Quantization bits (4 or 8) or None
        """
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=Flux2WeightDefinition,
            models={
                "vae": model.vae,
                "transformer": model.transformer,
                "text_encoder": model.text_encoder,
            },
        )

    @staticmethod
    def _apply_lora(model, lora_paths: list[str] | None, lora_scales: list[float] | None) -> None:
        """Apply LoRA adapters to the transformer.

        Args:
            model: Model instance with transformer
            lora_paths: Optional list of LoRA adapter paths
            lora_scales: Optional list of scales (one per adapter)
        """
        model.lora_paths, model.lora_scales = LoRALoader.load_and_apply_lora(
            lora_mapping=Flux2LoRAMapping.get_mapping(),
            transformer=model.transformer,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )
