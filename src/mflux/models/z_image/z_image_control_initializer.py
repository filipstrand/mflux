import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.resolution.quantization_resolution import QuantizationResolution
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.z_image.model.z_image_text_encoder.text_encoder import TextEncoder
from mflux.models.z_image.model.z_image_transformer.control_transformer import ZImageControlTransformer
from mflux.models.z_image.model.z_image_vae.vae import VAE
from mflux.models.z_image.weights.z_image_lora_mapping import ZImageLoRAMapping
from mflux.models.z_image.weights.z_image_weight_definition import ZImageWeightDefinition


class ZImageControlInitializer:
    """Initializer for the Z-Image strict pose ControlNet variant (sc-2257).

    Identical to :class:`ZImageInitializer` except the transformer is a
    :class:`ZImageControlTransformer` and the Fun-Controlnet-Union safetensors is
    overlaid onto the control branch.

    Quantization ordering matters: base weights and the control overlay are both
    applied at full precision FIRST, and the whole transformer is quantized once
    AFTERWARDS. Quantizing before the overlay would replace the control branch's
    ``nn.Linear`` modules with ``QuantizedLinear`` (derived from random init),
    which then can't accept the raw bf16 Fun-Controlnet weights. Applying weights
    first means every control Linear quantizes from its real trained weights, so
    ``quantize=8`` works end to end (≈ halves transformer memory vs bf16).
    """

    @staticmethod
    def init(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        control_weights_path: str,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ) -> None:
        path = model_path if model_path else model_config.model_name
        ZImageControlInitializer._init_config(model, model_config)
        weights = ZImageControlInitializer._load_weights(path)
        ZImageControlInitializer._init_tokenizers(model, path)
        ZImageControlInitializer._init_models(model)
        # Base + control weights at full precision, THEN quantize together.
        ZImageControlInitializer._set_base_weights(model, weights)
        ZImageControlInitializer._apply_control_weights(model, control_weights_path)
        model.bits = ZImageControlInitializer._quantize(model, quantize)
        ZImageControlInitializer._apply_lora(model, lora_paths, lora_scales)

    @staticmethod
    def _init_config(model, model_config: ModelConfig) -> None:
        from mflux.callbacks.callback_registry import CallbackRegistry

        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = None

    @staticmethod
    def _load_weights(model_path: str) -> LoadedWeights:
        return WeightLoader.load(weight_definition=ZImageWeightDefinition, model_path=model_path)

    @staticmethod
    def _init_tokenizers(model, model_path: str) -> None:
        model.tokenizers = TokenizerLoader.load_all(
            definitions=ZImageWeightDefinition.get_tokenizers(),
            model_path=model_path,
        )

    @staticmethod
    def _init_models(model) -> None:
        model.vae = VAE()
        model.transformer = ZImageControlTransformer()
        model.text_encoder = TextEncoder()

    @staticmethod
    def _set_base_weights(model, weights: LoadedWeights) -> None:
        # quantize_arg=None → WeightApplier only sets the weights (no quantization
        # yet); the control overlay + quantization happen afterwards.
        WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=None,
            weight_definition=ZImageWeightDefinition,
            models={
                "vae": model.vae,
                "transformer": model.transformer,
                "text_encoder": model.text_encoder,
            },
        )

    @staticmethod
    def _apply_control_weights(model, control_weights_path: str) -> None:
        # The Fun-Controlnet-Union safetensors keys (control_layers.*,
        # control_noise_refiner.*, control_all_x_embedder.*) map 1:1 onto the
        # ZImageControlTransformer param tree, so a direct tree_unflatten + update
        # is sufficient. Cast to the model precision (CN ships bf16) to match.
        raw = mx.load(str(control_weights_path))
        cast = [(k, v.astype(ModelConfig.precision)) for k, v in raw.items()]
        model.transformer.update(tree_unflatten(cast))
        mx.eval(model.transformer.parameters())

    @staticmethod
    def _quantize(model, quantize: int | None) -> int | None:
        # Quantize base + control together (mirrors WeightApplier._quantize but
        # runs after the control overlay so the control Linears quantize from
        # their real weights). stored=None because the diffusers source weights
        # are full precision.
        bits, warning = QuantizationResolution.resolve(stored=None, requested=quantize)
        if warning:
            print(f"⚠️  {warning}")
        if bits is None:
            return None

        group_size = 64

        def predicate(path, module) -> bool:
            # nn.quantize requires the weight's last dim divisible by the group
            # size (64). The base model's quantizable Linears all satisfy this,
            # but the control patch embedder is 33*4=132 wide → leave it (and any
            # other non-divisible Linear) in full precision. It's tiny, so the
            # memory cost is negligible.
            if not ZImageWeightDefinition.quantization_predicate(path, module):
                return False
            weight = getattr(module, "weight", None)
            return weight is not None and weight.shape[-1] % group_size == 0

        components = {c.name: c for c in ZImageWeightDefinition.get_components()}
        for name, module in (("vae", model.vae), ("transformer", model.transformer), ("text_encoder", model.text_encoder)):
            component = components.get(name)
            if component is not None and component.skip_quantization:
                continue
            nn.quantize(module, class_predicate=predicate, group_size=group_size, bits=bits)
        mx.eval(model.transformer.parameters(), model.vae.parameters(), model.text_encoder.parameters())
        return bits

    @staticmethod
    def _apply_lora(model, lora_paths: list[str] | None, lora_scales: list[float] | None) -> None:
        model.lora_paths, model.lora_scales = LoRALoader.load_and_apply_lora(
            lora_mapping=ZImageLoRAMapping.get_mapping(),
            transformer=model.transformer,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )
