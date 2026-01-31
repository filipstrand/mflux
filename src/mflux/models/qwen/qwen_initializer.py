from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.qwen.model.qwen_text_encoder.qwen_text_encoder import QwenTextEncoder
from mflux.models.qwen.model.qwen_text_encoder.qwen_vision_language_encoder import QwenVisionLanguageEncoder
from mflux.models.qwen.model.qwen_text_encoder.qwen_vision_transformer import VisionTransformer
from mflux.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer
from mflux.models.qwen.model.qwen_transformer.qwen_transformer_layered import QwenTransformerLayered
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
from mflux.models.qwen.model.qwen_vae.qwen_vae_rgba import QwenVAERGBA
from mflux.models.qwen.tokenizer.qwen_vision_language_processor import QwenVisionLanguageProcessor
from mflux.models.qwen.tokenizer.qwen_vision_language_tokenizer import QwenVisionLanguageTokenizer
from mflux.models.qwen.weights.qwen_lora_mapping import QwenLoRAMapping
from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition


class QwenImageInitializer:
    @staticmethod
    def _init_common(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        model_init_fn,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        add_vision_language: bool = False,
    ) -> None:
        """Common initialization logic shared across model variants.

        Args:
            model: Model instance to initialize
            model_config: Model configuration
            quantize: Quantization bits (None for full precision)
            model_init_fn: Function to call for model-specific initialization
            model_path: Optional custom model path
            lora_paths: Optional LoRA paths
            lora_scales: Optional LoRA scales
            add_vision_language: Whether to add vision-language tokenizer
        """
        path = model_path if model_path else model_config.model_name
        QwenImageInitializer._init_config(model, model_config)
        weights = QwenImageInitializer._load_weights(path)
        QwenImageInitializer._init_tokenizers(model, path)
        model_init_fn(model)
        QwenImageInitializer._apply_weights(model, weights, quantize)
        QwenImageInitializer._apply_lora(model, lora_paths, lora_scales)

        if add_vision_language:
            QwenImageInitializer._add_vision_language_encoder(model)

    @staticmethod
    def _add_vision_language_encoder(model) -> None:
        """Add vision-language tokenizer and encoder to model."""
        raw_tokenizer = model.tokenizers["qwen"].tokenizer
        processor = QwenVisionLanguageProcessor(tokenizer=raw_tokenizer)
        model.tokenizers["qwen_vl"] = QwenVisionLanguageTokenizer(
            processor=processor,
            max_length=1024,
            use_picture_prefix=True,
        )
        model.qwen_vl_encoder = QwenVisionLanguageEncoder(encoder=model.text_encoder.encoder)

    @staticmethod
    def init(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ) -> None:
        QwenImageInitializer._init_common(
            model,
            model_config,
            quantize,
            QwenImageInitializer._init_models,
            model_path,
            lora_paths,
            lora_scales,
            add_vision_language=False,
        )

    @staticmethod
    def init_edit(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ) -> None:
        QwenImageInitializer._init_common(
            model,
            model_config,
            quantize,
            QwenImageInitializer._init_edit_models,
            model_path,
            lora_paths,
            lora_scales,
            add_vision_language=True,
        )

    @staticmethod
    def init_layered(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ) -> None:
        """Initialize a Qwen-Image-Layered model for image decomposition.

        This variant uses:
        - RGBA VAE (4-channel input/output)
        - Layered transformer with additional timestep conditioning
        - Vision-language encoder for image+text conditioning
        """
        QwenImageInitializer._init_common(
            model,
            model_config,
            quantize,
            QwenImageInitializer._init_layered_models,
            model_path,
            lora_paths,
            lora_scales,
            add_vision_language=True,
        )

    @staticmethod
    def _init_config(model, model_config: ModelConfig) -> None:
        model.prompt_cache = {}
        model.model_config = model_config
        model.callbacks = CallbackRegistry()

        # OPTIMIZATION: Tiling disabled for optimal performance on high-RAM systems (Phase 4.3)
        # HIGH PRIORITY FIX: Detect available RAM before disabling tiling to prevent OOM
        import subprocess

        try:
            result = subprocess.run(["sysctl", "hw.memsize"], capture_output=True, text=True, check=True)
            mem_bytes = int(result.stdout.split(":")[1].strip())
            mem_gb = mem_bytes / (1024**3)

            # Only disable tiling if we have sufficient RAM (>= 128GB)
            # With 512GB RAM, we can process 2048x2048+ without tiling
            # With <128GB RAM, use default tiling config to prevent OOM
            if mem_gb >= 128:
                model.tiling_config = None  # Tiling disabled for maximum speed
            else:
                # Use default tiling config for systems with less RAM
                from mflux.models.common.vae.tiling_config import TilingConfig

                model.tiling_config = TilingConfig()
        except (subprocess.CalledProcessError, ValueError, IndexError, FileNotFoundError):
            # HIGH PRIORITY FIX: Added FileNotFoundError for cross-platform compatibility
            # On error (e.g., non-macOS system, missing sysctl), use safe default (tiling enabled)
            from mflux.models.common.vae.tiling_config import TilingConfig

            model.tiling_config = TilingConfig()

    @staticmethod
    def _load_weights(model_path: str) -> LoadedWeights:
        return WeightLoader.load(
            weight_definition=QwenWeightDefinition,
            model_path=model_path,
        )

    @staticmethod
    def _init_tokenizers(model, model_path: str) -> None:
        model.tokenizers = TokenizerLoader.load_all(
            definitions=QwenWeightDefinition.get_tokenizers(),
            model_path=model_path,
        )

    @staticmethod
    def _init_models(model) -> None:
        model.vae = QwenVAE()
        model.transformer = QwenTransformer()
        model.text_encoder = QwenTextEncoder()

    @staticmethod
    def _init_edit_models(model) -> None:
        model.vae = QwenVAE()
        model.transformer = QwenTransformer()
        model.text_encoder = QwenTextEncoder()
        model.text_encoder.encoder.visual = VisionTransformer()

    @staticmethod
    def _init_layered_models(model) -> None:
        """Initialize models for Qwen-Image-Layered with RGBA VAE and layered transformer."""
        model.vae = QwenVAERGBA()
        model.transformer = QwenTransformerLayered()
        model.text_encoder = QwenTextEncoder()
        model.text_encoder.encoder.visual = VisionTransformer()

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None) -> None:
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=QwenWeightDefinition,
            models={
                "vae": model.vae,
                "transformer": model.transformer,
                "text_encoder": model.text_encoder,
            },
        )

    @staticmethod
    def _apply_lora(model, lora_paths: list[str] | None, lora_scales: list[float] | None) -> None:
        model.lora_paths, model.lora_scales = LoRALoader.load_and_apply_lora(
            lora_mapping=QwenLoRAMapping.get_mapping(),
            transformer=model.transformer,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )
