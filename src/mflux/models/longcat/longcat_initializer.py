"""
LongCat Model Initializer.

Handles loading weights, tokenizers, and initializing all model components.
LongCat-Image uses:
- Qwen2.5-VL text encoder (3584 hidden, 28 layers)
- LongCat Flow Match transformer (10 joint + 20 single blocks)
- Standard 16-channel VAE (same as FLUX)
"""

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.flux.model.flux_vae.vae import VAE
from mflux.models.longcat.model.longcat_text_encoder.longcat_text_encoder import LongCatTextEncoder
from mflux.models.longcat.model.longcat_transformer.longcat_transformer import LongCatTransformer
from mflux.models.longcat.weights.longcat_lora_mapping import LongCatLoRAMapping
from mflux.models.longcat.weights.longcat_weight_definition import LongCatWeightDefinition


class LongCatInitializer:
    """
    Initializer for LongCat-Image model.

    Key differences from other models:
    1. Uses Qwen2.5-VL as text encoder (3584 hidden, 28 layers)
    2. Uses LongCatTransformer (10 joint + 20 single blocks)
    3. Uses standard FLUX VAE (16 latent channels)
    4. No guidance embeddings (guidance_embeds: false)
    """

    @staticmethod
    def init(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ) -> None:
        """
        Initialize the LongCat model.

        Args:
            model: The LongCat model instance to initialize
            model_config: Model configuration
            quantize: Quantization bit width (4 or 8) or None for full precision
            model_path: Path to model weights (local or HuggingFace repo ID)
            lora_paths: List of LoRA paths (local files or HuggingFace repos)
            lora_scales: List of LoRA scale factors (default 1.0)

        Raises:
            TypeError: If model_config is not a ModelConfig instance
            ValueError: If quantize is not 4, 8, or None
            ValueError: If lora_paths and lora_scales lengths don't match
        """
        # Validate inputs early to catch configuration errors
        LongCatInitializer._validate_inputs(model_config, quantize, lora_paths, lora_scales)

        path = model_path if model_path else model_config.model_name

        LongCatInitializer._init_config(model, model_config)
        weights = LongCatInitializer._load_weights(path)
        LongCatInitializer._init_tokenizers(model, path, model_config)
        LongCatInitializer._init_models(model)
        LongCatInitializer._apply_weights(model, weights, quantize)
        LongCatInitializer._apply_lora(model, lora_paths, lora_scales)

    @staticmethod
    def _validate_inputs(
        model_config: ModelConfig,
        quantize: int | None,
        lora_paths: list[str] | None,
        lora_scales: list[float] | None,
    ) -> None:
        """
        Validate initialization inputs to catch configuration errors early.

        Args:
            model_config: Model configuration to validate
            quantize: Quantization bit width to validate
            lora_paths: LoRA paths to validate
            lora_scales: LoRA scales to validate

        Raises:
            TypeError: If model_config is not a ModelConfig instance or lacks required attributes
            ValueError: If quantize is not 4, 8, or None
            ValueError: If lora_paths and lora_scales lengths don't match
        """
        # Validate model_config type
        if not isinstance(model_config, ModelConfig):
            raise TypeError(
                f"model_config must be a ModelConfig instance, got {type(model_config).__name__}"
            )

        # Validate model_config has required attributes
        required_attrs = ["model_name", "max_sequence_length"]
        missing_attrs = [attr for attr in required_attrs if not hasattr(model_config, attr)]
        if missing_attrs:
            raise TypeError(
                f"model_config is missing required attributes: {', '.join(missing_attrs)}"
            )

        # Validate quantize value
        if quantize is not None and quantize not in (4, 8):
            raise ValueError(
                f"quantize must be 4, 8, or None for full precision, got {quantize}"
            )

        # Validate lora_paths and lora_scales match in length if both provided
        if lora_paths is not None and lora_scales is not None:
            if len(lora_paths) != len(lora_scales):
                raise ValueError(
                    f"lora_paths and lora_scales must have the same length. "
                    f"Got {len(lora_paths)} paths and {len(lora_scales)} scales"
                )

    @staticmethod
    def _init_config(model, model_config: ModelConfig) -> None:
        """Initialize model configuration and callbacks."""
        model.prompt_cache = {}
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = None
        model.lora_paths = None
        model.lora_scales = None

    @staticmethod
    def _load_weights(model_path: str) -> LoadedWeights:
        """Load model weights from path or HuggingFace."""
        return WeightLoader.load(
            weight_definition=LongCatWeightDefinition,
            model_path=model_path,
        )

    @staticmethod
    def _init_tokenizers(model, model_path: str, model_config: ModelConfig) -> None:
        """
        Initialize tokenizers.

        For LongCat, uses Qwen2 tokenizer for the Qwen2.5-VL text encoder.
        """
        max_length_overrides = (
            {"qwen": model_config.max_sequence_length} if model_config.max_sequence_length is not None else {}
        )
        model.tokenizers = TokenizerLoader.load_all(
            definitions=LongCatWeightDefinition.get_tokenizers(),
            model_path=model_path,
            max_length_overrides=max_length_overrides,
        )

    @staticmethod
    def _init_models(model) -> None:
        """
        Initialize model components.

        For LongCat:
        - VAE (standard 16-channel, same as FLUX)
        - Qwen2.5-VL text encoder (3584 hidden, 28 layers)
        - LongCatTransformer (10 joint + 20 single blocks)
        """
        model.vae = VAE()
        model.text_encoder = LongCatTextEncoder()
        model.transformer = LongCatTransformer()

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None) -> None:
        """Apply loaded weights to model components and optionally quantize."""
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=LongCatWeightDefinition,
            models={
                "vae": model.vae,
                "transformer": model.transformer,
                "text_encoder": model.text_encoder,
            },
        )

    @staticmethod
    def _apply_lora(
        model,
        lora_paths: list[str] | None,
        lora_scales: list[float] | None,
    ) -> None:
        """
        Apply LoRA weights to the model.

        Args:
            model: The LongCat model instance
            lora_paths: List of LoRA paths (local files or HuggingFace repos)
            lora_scales: List of LoRA scale factors
        """
        model.lora_paths, model.lora_scales = LoRALoader.load_and_apply_lora(
            lora_mapping=LongCatLoRAMapping.get_mapping(),
            transformer=model.transformer,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )
