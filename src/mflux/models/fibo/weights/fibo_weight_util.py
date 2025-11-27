from typing import TYPE_CHECKING

import mlx.nn as nn

from mflux.models.common.quantization.quantization_util import QuantizationUtil

if TYPE_CHECKING:
    from mflux.models.fibo.weights.fibo_weight_handler import FIBOWeightHandler


class FIBOWeightUtil:
    @staticmethod
    def set_weights_and_quantize(
        quantize_arg: int | None,
        weights: "FIBOWeightHandler",
        vae: nn.Module,
        transformer: nn.Module,
        text_encoder: nn.Module | None = None,
    ) -> int | None:
        if weights.meta_data.quantization_level is None and quantize_arg is None:
            FIBOWeightUtil._set_model_weights(weights, vae, transformer, text_encoder)
            return None

        if weights.meta_data.quantization_level is None and quantize_arg is not None:
            bits = quantize_arg
            FIBOWeightUtil._set_model_weights(weights, vae, transformer, text_encoder)
            QuantizationUtil.quantize_fibo_models(vae, transformer, text_encoder, bits, weights)
            return bits

        if weights.meta_data.quantization_level is not None:
            bits = weights.meta_data.quantization_level
            QuantizationUtil.quantize_fibo_models(vae, transformer, text_encoder, bits, weights)
            FIBOWeightUtil._set_model_weights(weights, vae, transformer, text_encoder)
            return bits

        raise Exception("Error setting weights")

    @staticmethod
    def _set_model_weights(
        weights: "FIBOWeightHandler",
        vae: nn.Module,
        transformer: nn.Module,
        text_encoder: nn.Module | None = None,
    ):
        vae.update(weights.vae, strict=False)
        transformer.update(weights.transformer, strict=False)
        text_encoder.update(weights.text_encoder, strict=False)
