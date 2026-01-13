"""
NewBie-image Model Initializer.

Handles loading weights, tokenizers, and initializing all model components.
"""

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.flux.model.flux_vae.vae import VAE
from mflux.models.newbie.model.newbie_text_encoder.gemma3_encoder import Gemma3Encoder
from mflux.models.newbie.model.newbie_text_encoder.jina_clip_encoder import JinaCLIPEncoder
from mflux.models.newbie.model.newbie_transformer.nextdit import NextDiT
from mflux.models.newbie.weights.newbie_lora_mapping import NewBieLoRAMapping
from mflux.models.newbie.weights.newbie_weight_definition import NewBieWeightDefinition


class NewBieInitializer:
    """
    Initializer for NewBie-image model.

    Components:
    - VAE: FLUX.1-dev VAE (16 channels)
    - Gemma3 encoder: Gemma3-4B-it (2560 dim)
    - Jina CLIP encoder: Jina CLIP v2 (1024 dim)
    - Transformer: NextDiT (36 blocks, GQA)
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
        Initialize the NewBie-image model.

        Args:
            model: The NewBie model instance to initialize
            model_config: Model configuration
            quantize: Quantization bit width (4 or 8) or None for full precision
            model_path: Path to model weights (local or HuggingFace repo ID)
            lora_paths: List of LoRA paths (local files or HuggingFace repos)
            lora_scales: List of LoRA scale factors (default 1.0)
        """
        path = model_path if model_path else model_config.model_name

        NewBieInitializer._init_config(model, model_config)
        weights = NewBieInitializer._load_weights(path)
        NewBieInitializer._init_tokenizers(model, path, model_config)
        NewBieInitializer._init_models(model, model_config, weights)
        NewBieInitializer._apply_weights(model, weights, quantize)
        NewBieInitializer._apply_lora(model, lora_paths, lora_scales)

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
            weight_definition=NewBieWeightDefinition,
            model_path=model_path,
        )

    @staticmethod
    def _init_tokenizers(model, model_path: str, model_config: ModelConfig) -> None:
        """
        Initialize tokenizers.

        For NewBie, both Gemma3 and Jina CLIP tokenizers are needed.
        """
        max_length_overrides = {}
        if model_config.max_sequence_length is not None:
            max_length_overrides["gemma3"] = model_config.max_sequence_length

        model.tokenizers = TokenizerLoader.load_all(
            definitions=NewBieWeightDefinition.get_tokenizers(),
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

        For NewBie:
        - VAE (FLUX.1-dev, 16-channel)
        - Gemma3 encoder (2560 dim)
        - Jina CLIP encoder (1024 dim)
        - NextDiT transformer (36 blocks, GQA)
        """
        model.vae = VAE()
        model.gemma3_text_encoder = Gemma3Encoder()
        model.jina_clip_encoder = JinaCLIPEncoder()

        # Get number of blocks from weights, fallback to default 36 for NewBie-image
        num_blocks = weights.num_dit_blocks()
        if num_blocks == 0:
            # Weights don't contain block count, use default from model config or 36
            num_blocks = getattr(model_config, 'num_dit_blocks', 36)

        model.transformer = NextDiT(
            model_config=model_config,
            num_blocks=num_blocks,
        )

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None) -> None:
        """Apply loaded weights to model components and optionally quantize."""
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=NewBieWeightDefinition,
            models={
                "vae": model.vae,
                "transformer": model.transformer,
                "gemma3_encoder": model.gemma3_text_encoder,
                "jina_clip_encoder": model.jina_clip_encoder,
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

        Uses NewBieLoRAMapping for NextDiT-style GQA attention layers.

        Args:
            model: The NewBie model instance
            lora_paths: List of LoRA paths (local files or HuggingFace repos)
            lora_scales: List of LoRA scale factors
        """
        model.lora_paths, model.lora_scales = LoRALoader.load_and_apply_lora(
            lora_mapping=NewBieLoRAMapping.get_mapping(),
            transformer=model.transformer,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )
