"""
Chroma Model Initializer.

Handles loading weights, tokenizers, and initializing all model components.
Key difference from FluxInitializer: no CLIP encoder, uses ChromaWeightDefinition.
"""

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.chroma.model.chroma_transformer.chroma_transformer import ChromaTransformer
from mflux.models.chroma.weights.chroma_weight_definition import ChromaWeightDefinition
from mflux.models.common.config import ModelConfig
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.flux.model.flux_text_encoder.t5_encoder.t5_encoder import T5Encoder
from mflux.models.flux.model.flux_vae.vae import VAE


class ChromaInitializer:
    """
    Initializer for Chroma model.

    Key differences from FluxInitializer:
    1. Uses ChromaWeightDefinition (T5 encoder only, no CLIP)
    2. Uses ChromaTransformer (DistilledGuidanceLayer instead of TimeTextEmbed)
    3. No CLIP encoder initialization
    4. No LoRA support (initial version)
    """

    @staticmethod
    def init(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None = None,
    ) -> None:
        """
        Initialize the Chroma model.

        Args:
            model: The Chroma model instance to initialize
            model_config: Model configuration
            quantize: Quantization bit width (4 or 8) or None for full precision
            model_path: Path to model weights (local or HuggingFace repo ID)
        """
        path = model_path if model_path else model_config.model_name

        ChromaInitializer._init_config(model, model_config)
        weights = ChromaInitializer._load_weights(path)
        ChromaInitializer._init_tokenizers(model, path, model_config)
        ChromaInitializer._init_models(model, model_config, weights)
        ChromaInitializer._apply_weights(model, weights, quantize)

    @staticmethod
    def _init_config(model, model_config: ModelConfig) -> None:
        """Initialize model configuration and callbacks."""
        model.prompt_cache = {}
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = None
        # No LoRA paths for initial version
        model.lora_paths = None
        model.lora_scales = None

    @staticmethod
    def _load_weights(model_path: str) -> LoadedWeights:
        """Load model weights from path or HuggingFace."""
        return WeightLoader.load(
            weight_definition=ChromaWeightDefinition,
            model_path=model_path,
        )

    @staticmethod
    def _init_tokenizers(model, model_path: str, model_config: ModelConfig) -> None:
        """
        Initialize tokenizers.

        For Chroma, only T5 tokenizer is needed (no CLIP).
        """
        max_length_overrides = (
            {"t5": model_config.max_sequence_length} if model_config.max_sequence_length is not None else {}
        )
        model.tokenizers = TokenizerLoader.load_all(
            definitions=ChromaWeightDefinition.get_tokenizers(),
            model_path=model_path,
            max_length_overrides=max_length_overrides,
        )

    @staticmethod
    def _init_models(
        model,
        model_config: ModelConfig,
        weights: LoadedWeights,
    ) -> None:
        """
        Initialize model components.

        For Chroma:
        - VAE (same as FLUX)
        - T5 encoder (same as FLUX)
        - ChromaTransformer (different from FLUX)
        - NO CLIP encoder
        """
        model.vae = VAE()
        model.t5_text_encoder = T5Encoder()
        # No CLIP encoder for Chroma

        model.transformer = ChromaTransformer(
            model_config=model_config,
            num_transformer_blocks=weights.num_transformer_blocks(),
            num_single_transformer_blocks=weights.num_single_transformer_blocks(),
        )

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None) -> None:
        """Apply loaded weights to model components and optionally quantize."""
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=ChromaWeightDefinition,
            models={
                "vae": model.vae,
                "transformer": model.transformer,
                "t5_encoder": model.t5_text_encoder,
                # No CLIP encoder for Chroma
            },
        )
