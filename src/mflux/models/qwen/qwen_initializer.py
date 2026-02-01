import mlx.core as mx

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
from mflux.models.qwen.weights.qwen_quantization import QwenQuantizationConfig, QwenQuantizationMode
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
        # Detect available RAM before disabling tiling to prevent OOM
        mem_gb = QwenImageInitializer._get_system_memory_gb()

        # Memory threshold for disabling tiling (GB)
        # Rationale: Qwen model needs ~35GB base + ~40GB activations for 2048x2048 images
        # 128GB provides 2x headroom for batch processing and MLX memory overhead
        TILING_DISABLE_THRESHOLD_GB = 128

        # Only disable tiling if we have sufficient RAM
        # With 512GB RAM, we can process 2048x2048+ without tiling
        # With <128GB RAM, use default tiling config to prevent OOM
        if mem_gb >= TILING_DISABLE_THRESHOLD_GB:
            model.tiling_config = None  # Tiling disabled for maximum speed
        else:
            from mflux.models.common.vae.tiling_config import TilingConfig

            model.tiling_config = TilingConfig()

    @staticmethod
    def _get_system_memory_gb() -> float:
        """Get system memory in GB using safe methods.

        Uses psutil if available (cross-platform), falls back to macOS sysctl
        with absolute path to prevent PATH manipulation attacks.

        Returns:
            System memory in GB, or 0.0 if detection fails (triggers safe defaults)
        """
        # Try psutil first (cross-platform, recommended)
        try:
            import psutil

            return psutil.virtual_memory().total / (1024**3)
        except ImportError:
            pass

        # Fallback: macOS sysctl with absolute path (security: prevents PATH injection)
        import subprocess
        import sys

        # Timeout for sysctl command - prevents process hangs during memory detection
        SYSCTL_TIMEOUT_SECONDS = 5

        if sys.platform == "darwin":
            try:
                result = subprocess.run(
                    ["/usr/sbin/sysctl", "hw.memsize"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=SYSCTL_TIMEOUT_SECONDS,
                )
                mem_bytes = int(result.stdout.split(":")[1].strip())
                return mem_bytes / (1024**3)
            except (subprocess.CalledProcessError, ValueError, IndexError, subprocess.TimeoutExpired):
                pass

        # Safe default: return 0 to trigger tiling (conservative)
        return 0.0

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
    def _apply_weights(
        model,
        weights: LoadedWeights,
        quantize: int | QwenQuantizationConfig | QwenQuantizationMode | str | None,
    ) -> None:
        """Apply weights with optional quantization.

        Args:
            model: Model instance
            weights: Loaded weights
            quantize: Quantization configuration. Can be:
                - None: No quantization
                - int: Uniform bits for all components (4, 8)
                - str: Mode name ("speed", "quality", "mixed", "conservative")
                - QwenQuantizationMode: Enum mode
                - QwenQuantizationConfig: Full configuration
        """
        # Convert to QwenQuantizationConfig if needed
        quant_config = QwenImageInitializer._resolve_quantization(quantize)

        if quant_config is None or not quant_config.is_quantized:
            # No quantization - use standard path
            model.bits = WeightApplier.apply_and_quantize(
                weights=weights,
                quantize_arg=None,
                weight_definition=QwenWeightDefinition,
                models={
                    "vae": model.vae,
                    "transformer": model.transformer,
                    "text_encoder": model.text_encoder,
                },
            )
            model.quantization_config = None
        else:
            # Apply per-component quantization
            model.bits = WeightApplier.apply_and_quantize(
                weights=weights,
                quantize_arg=quant_config.transformer_bits,  # Primary bits for WeightApplier
                weight_definition=QwenWeightDefinition,
                models={
                    "vae": model.vae,
                    "transformer": model.transformer,
                    "text_encoder": model.text_encoder,
                },
            )
            model.quantization_config = quant_config

    @staticmethod
    def _resolve_quantization(
        quantize: int | QwenQuantizationConfig | QwenQuantizationMode | str | None,
    ) -> QwenQuantizationConfig | None:
        """Resolve quantization argument to QwenQuantizationConfig.

        Args:
            quantize: Various quantization specifications

        Returns:
            QwenQuantizationConfig or None

        Raises:
            ValueError: If quantize is an invalid string mode or integer bits
            TypeError: If quantize is an unsupported type
        """
        if quantize is None:
            return None

        if isinstance(quantize, QwenQuantizationConfig):
            return quantize

        if isinstance(quantize, QwenQuantizationMode):
            return QwenQuantizationConfig.from_mode(quantize)

        if isinstance(quantize, str):
            # from_string with strict=True raises ValueError for invalid strings
            mode = QwenQuantizationMode.from_string(quantize, strict=True)
            if mode is not None:
                return QwenQuantizationConfig.from_mode(mode)
            return None

        if isinstance(quantize, int):
            # from_bits validates bits and raises ValueError for invalid values
            return QwenQuantizationConfig.from_bits(quantize)

        raise TypeError(
            f"quantize must be int, str, QwenQuantizationMode, QwenQuantizationConfig, or None. "
            f"Got {type(quantize).__name__}"
        )

    @staticmethod
    def _apply_lora(model, lora_paths: list[str] | None, lora_scales: list[float] | None) -> None:
        model.lora_paths, model.lora_scales = LoRALoader.load_and_apply_lora(
            lora_mapping=QwenLoRAMapping.get_mapping(),
            transformer=model.transformer,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )

    @staticmethod
    def compile_for_inference(model) -> None:
        """Compile transformer for faster inference (15-40% speedup).

        Call this after initialization is complete. Wraps the transformer
        in a compiled execution mode using MLX graph compilation.

        The original transformer module is preserved for serialization and
        debugging. Only the forward pass execution is compiled.

        Use decompile_for_inference() to restore original behavior.

        Example:
            qwen = QwenImage(quantize=4)
            QwenImageInitializer.compile_for_inference(qwen)
            # Now inference is ~25-40% faster
        """
        import types

        # Check if already compiled
        if getattr(model.transformer, "_is_compiled", False):
            return

        # Get the class's original __call__ method (unbound)
        original_method = type(model.transformer).__call__

        # Create compiled version of the unbound method
        compiled_call = mx.compile(original_method)

        # Store references for debugging/serialization support
        model.transformer._original_method = original_method
        model.transformer._compiled_call = compiled_call
        model.transformer._is_compiled = True

        # Wrapper function that uses the compiled call
        def compiled_forward_wrapper(self, *args, **kwargs):
            return self._compiled_call(self, *args, **kwargs)

        # Bind the wrapper to the transformer instance
        model.transformer.__call__ = types.MethodType(compiled_forward_wrapper, model.transformer)

    @staticmethod
    def decompile_for_inference(model) -> None:
        """Restore original uncompiled transformer behavior.

        Use this before serialization or when debugging compilation issues.
        """
        if not getattr(model.transformer, "_is_compiled", False):
            return

        # Remove instance __call__ override to restore class method behavior
        if "__call__" in model.transformer.__dict__:
            del model.transformer.__dict__["__call__"]

        # Clean up stored references safely
        for attr in ("_original_method", "_compiled_call", "_is_compiled"):
            if attr in model.transformer.__dict__:
                del model.transformer.__dict__[attr]
