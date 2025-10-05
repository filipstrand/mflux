from typing import TYPE_CHECKING

import mlx.nn as nn

if TYPE_CHECKING:
    from mflux.models.flux.variants.controlnet.weight_handler_controlnet import WeightHandlerControlnet
    from mflux.models.flux.weights.weight_handler import WeightHandler
    from mflux.models.qwen.weights.qwen_weight_handler import QwenWeightHandler


class QuantizationUtil:
    @staticmethod
    def quantize_model(
        vae: nn.Module,
        transformer: nn.Module,
        t5_text_encoder: nn.Module,
        clip_text_encoder: nn.Module,
        quantize: int,
        weights: "WeightHandler",
    ) -> None:
        q_level = weights.meta_data.quantization_level

        # mx.save_tensors saves metadata dict kv 'quantization_level': 'None' as a str: str mapping
        # we coerce both configs to NoneType to help users use non-quantized saved model files
        if q_level == "None":
            q_level = None
        if quantize == "None":
            quantize = None

        if quantize is not None or q_level is not None:
            bits = int(q_level) if q_level is not None else quantize
            nn.quantize(vae, bits=bits)
            nn.quantize(transformer, bits=bits)
            nn.quantize(t5_text_encoder, bits=bits)
            nn.quantize(clip_text_encoder, bits=bits)

    @staticmethod
    def quantize_controlnet(
        quantize: int,
        weights: "WeightHandlerControlnet",
        transformer_controlnet: nn.Module,
    ) -> None:
        q_level = weights.meta_data.quantization_level

        if quantize is not None or q_level is not None:
            bits = int(q_level) if q_level is not None else quantize
            nn.quantize(transformer_controlnet, bits=bits)

    @staticmethod
    def quantize_redux_models(
        quantize: int,
        weights: "WeightHandler",
        redux_encoder: nn.Module,
        siglip_vision_transformer: nn.Module,
    ) -> None:
        q_level = weights.meta_data.quantization_level

        if quantize is not None or q_level is not None:
            bits = int(q_level) if q_level is not None else quantize
            nn.quantize(redux_encoder, class_predicate=QuantizationUtil.quantization_predicate, bits=bits)
            nn.quantize(siglip_vision_transformer, class_predicate=QuantizationUtil.quantization_predicate, bits=bits)

    @staticmethod
    def quantization_predicate(path, m):
        # 1. Skip Conv2d layers
        if isinstance(m, nn.Conv2d):
            return False

        # 2. Skip any layer with incompatible dimensions
        if hasattr(m, "weight") and hasattr(m.weight, "shape"):
            if m.weight.shape == (1152, 4304):
                return False

            if m.weight.shape[-1] % 64 != 0:
                return False

        # Only quantize layers that have to_quantized method
        return hasattr(m, "to_quantized")

    @staticmethod
    def quantize_qwen_models(
        text_encoder: nn.Module,
        vae: nn.Module,
        transformer: nn.Module,
        quantize: int,
        weights: "QwenWeightHandler",
    ) -> None:
        q_level = weights.meta_data.quantization_level

        if quantize is not None or q_level is not None:
            bits = int(q_level) if q_level is not None else quantize
            nn.quantize(vae, bits=bits)
            nn.quantize(transformer, bits=bits)
            
            # For text encoder, exclude visual components from quantization
            # as they have dimensions not compatible with MLX quantization
            if hasattr(text_encoder, 'encoder') and hasattr(text_encoder.encoder, 'visual'):
                # Quantize everything except visual components
                nn.quantize(text_encoder.encoder.embed_tokens, bits=bits)
                # Quantize each layer individually since layers is a list
                for layer in text_encoder.encoder.layers:
                    nn.quantize(layer, bits=bits)
                nn.quantize(text_encoder.encoder.norm, bits=bits)
                # Skip text_encoder.encoder.visual - leave unquantized
                print(f"🔧 Quantized text encoder to {bits} bits (excluding visual components)")
            else:
                nn.quantize(text_encoder, bits=bits)
