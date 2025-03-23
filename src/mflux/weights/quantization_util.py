from typing import TYPE_CHECKING

import mlx.nn as nn

if TYPE_CHECKING:
    from mflux.controlnet.weight_handler_controlnet import WeightHandlerControlnet
    from mflux.weights.weight_handler import WeightHandler


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
