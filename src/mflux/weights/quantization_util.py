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
            # fmt: off
            nn.quantize(vae, class_predicate=lambda _, m: isinstance(m, nn.Linear), group_size=64, bits=bits)
            nn.quantize(transformer, class_predicate=lambda _, m: isinstance(m, nn.Linear) and len(m.weight[1]) > 64, group_size=64, bits=bits)
            nn.quantize(t5_text_encoder, class_predicate=lambda _, m: isinstance(m, nn.Linear), group_size=64, bits=bits)
            nn.quantize(clip_text_encoder, class_predicate=lambda _, m: isinstance(m, nn.Linear), group_size=64, bits=bits)
            # fmt: on

    @staticmethod
    def quantize_controlnet(
        quantize: int,
        weights: "WeightHandlerControlnet",
        transformer_controlnet: nn.Module,
    ):
        q_level = weights.meta_data.quantization_level
        if quantize is not None or q_level is not None:
            bits = int(q_level) if q_level is not None else quantize
            nn.quantize(transformer_controlnet, class_predicate=lambda _, m: isinstance(m, nn.Linear) and len(m.weight[1]) > 128, group_size=128, bits=bits)  # fmt: off
