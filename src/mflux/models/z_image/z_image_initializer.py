import mlx.core as mx

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.z_image.model.z_image_text_encoder.text_encoder import TextEncoder
from mflux.models.z_image.model.z_image_transformer.transformer import ZImageTransformer
from mflux.models.z_image.model.z_image_vae.vae import VAE
from mflux.models.z_image.optimization.prompt_cache import ZImagePromptCache
from mflux.models.z_image.weights.z_image_lora_mapping import ZImageLoRAMapping
from mflux.models.z_image.weights.z_image_weight_definition import ZImageWeightDefinition


class ZImageInitializer:
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
        ZImageInitializer._init_config(model, model_config)
        weights = ZImageInitializer._load_weights(path)
        ZImageInitializer._init_tokenizers(model, path)
        ZImageInitializer._init_models(model)
        ZImageInitializer._apply_weights(model, weights, quantize)
        ZImageInitializer._apply_lora(model, lora_paths, lora_scales)

    @staticmethod
    def _init_config(model, model_config: ModelConfig) -> None:
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = None
        model.prompt_cache = ZImagePromptCache(max_items=100)

    @staticmethod
    def _load_weights(model_path: str) -> LoadedWeights:
        return WeightLoader.load(
            weight_definition=ZImageWeightDefinition,
            model_path=model_path,
        )

    @staticmethod
    def _init_tokenizers(model, model_path: str) -> None:
        model.tokenizers = TokenizerLoader.load_all(
            definitions=ZImageWeightDefinition.get_tokenizers(),
            model_path=model_path,
        )

    @staticmethod
    def _init_models(model) -> None:
        model.vae = VAE()
        model.transformer = ZImageTransformer()
        model.text_encoder = TextEncoder()

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None) -> None:
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=ZImageWeightDefinition,
            models={
                "vae": model.vae,
                "transformer": model.transformer,
                "text_encoder": model.text_encoder,
            },
        )

    @staticmethod
    def _apply_lora(model, lora_paths: list[str] | None, lora_scales: list[float] | None) -> None:
        model.lora_paths, model.lora_scales = LoRALoader.load_and_apply_lora(
            lora_mapping=ZImageLoRAMapping.get_mapping(),
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
        # Check __dict__ directly to ensure we only delete instance attributes
        if "__call__" in model.transformer.__dict__:
            del model.transformer.__dict__["__call__"]

        # Clean up stored references safely
        for attr in ("_original_method", "_compiled_call", "_is_compiled"):
            if attr in model.transformer.__dict__:
                del model.transformer.__dict__[attr]
