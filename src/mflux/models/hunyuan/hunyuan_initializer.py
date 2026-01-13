"""
Hunyuan-DiT Model Initializer.

Handles loading weights, tokenizers, and initializing all model components.
"""

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.flux.model.flux_text_encoder.clip_encoder.clip_encoder import CLIPEncoder
from mflux.models.flux.model.flux_text_encoder.t5_encoder.t5_encoder import T5Encoder
from mflux.models.flux.model.flux_vae.vae import VAE
from mflux.models.hunyuan.model.hunyuan_transformer.hunyuan_dit import HunyuanDiT
from mflux.models.hunyuan.weights.hunyuan_lora_mapping import HunyuanLoRAMapping
from mflux.models.hunyuan.weights.hunyuan_weight_definition import HunyuanWeightDefinition


class HunyuanInitializer:
    """
    Initializer for Hunyuan-DiT model.

    Components:
    - VAE: Standard 4-channel SDXL VAE
    - CLIP encoder: Chinese CLIP (1024 dim)
    - T5 encoder: mT5-XXL (2048 dim)
    - Transformer: HunyuanDiT2DModel (28 blocks)
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
        Initialize the Hunyuan-DiT model.

        Args:
            model: The Hunyuan model instance to initialize
            model_config: Model configuration
            quantize: Quantization bit width (4 or 8) or None for full precision
            model_path: Path to model weights (local or HuggingFace repo ID)
            lora_paths: List of LoRA paths (local files or HuggingFace repos)
            lora_scales: List of LoRA scale factors (default 1.0)
        """
        path = model_path if model_path else model_config.model_name

        HunyuanInitializer._init_config(model, model_config)
        weights = HunyuanInitializer._load_weights(path)
        HunyuanInitializer._init_tokenizers(model, path, model_config)
        HunyuanInitializer._init_models(model, model_config, weights)
        HunyuanInitializer._apply_weights(model, weights, quantize)
        HunyuanInitializer._apply_lora(model, lora_paths, lora_scales)

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
            weight_definition=HunyuanWeightDefinition,
            model_path=model_path,
        )

    @staticmethod
    def _init_tokenizers(model, model_path: str, model_config: ModelConfig) -> None:
        """
        Initialize tokenizers.

        For Hunyuan, both CLIP and T5 tokenizers are needed.
        """
        max_length_overrides = {}
        if model_config.max_sequence_length is not None:
            max_length_overrides["t5"] = model_config.max_sequence_length

        model.tokenizers = TokenizerLoader.load_all(
            definitions=HunyuanWeightDefinition.get_tokenizers(),
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

        For Hunyuan:
        - VAE (standard 4-channel)
        - CLIP encoder (Chinese CLIP, 1024 dim)
        - T5 encoder (mT5-XXL, 2048 dim)
        - HunyuanDiT transformer (28 blocks)
        """
        model.vae = VAE()
        model.clip_text_encoder = CLIPEncoder()
        model.t5_text_encoder = T5Encoder()

        model.transformer = HunyuanDiT(
            model_config=model_config,
            num_blocks=weights.num_dit_blocks() if hasattr(weights, 'num_dit_blocks') else 28,
        )

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None) -> None:
        """Apply loaded weights to model components and optionally quantize."""
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=HunyuanWeightDefinition,
            models={
                "vae": model.vae,
                "transformer": model.transformer,
                "clip_encoder": model.clip_text_encoder,
                "t5_encoder": model.t5_text_encoder,
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

        Uses HunyuanLoRAMapping for DiT-style attention layers.

        Args:
            model: The Hunyuan model instance
            lora_paths: List of LoRA paths (local files or HuggingFace repos)
            lora_scales: List of LoRA scale factors
        """
        model.lora_paths, model.lora_scales = LoRALoader.load_and_apply_lora(
            lora_mapping=HunyuanLoRAMapping.get_mapping(),
            transformer=model.transformer,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )
