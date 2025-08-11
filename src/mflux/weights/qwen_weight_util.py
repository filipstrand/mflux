from typing import TYPE_CHECKING

import mlx.nn as nn

from mflux.config.config import Config
from mflux.weights.quantization_util import QuantizationUtil

if TYPE_CHECKING:
    from mflux.weights.qwen_weight_handler import QwenImageWeightHandler

from mflux.qwen.qwen_transformer_full import QwenImageTransformerApplier


class QwenWeightUtil:
    @staticmethod
    def flatten(params):
        return [(k, v) for p in params for (k, v) in p]

    @staticmethod
    def reshape_weights(key, value):
        if len(value.shape) == 4:
            value = value.transpose(0, 2, 3, 1)
        value = value.reshape(-1).reshape(value.shape).astype(Config.precision)
        return [(key, value)]

    @staticmethod
    def set_weights_and_quantize(
        quantize_arg: int | None,
        weights: "QwenImageWeightHandler",
        vae: nn.Module,
        transformer: nn.Module,
    ) -> int | None:
        if weights.meta_data.quantization_level is None and quantize_arg is None:
            QwenWeightUtil._set_model_weights(weights, vae, transformer)
            return None

        if weights.meta_data.quantization_level is None and quantize_arg is not None:
            bits = quantize_arg
            QwenWeightUtil._set_model_weights(weights, vae, transformer)
            QuantizationUtil.quantize_qwen_models(vae, transformer, bits, weights)
            return bits

        if weights.meta_data.quantization_level is not None:
            bits = weights.meta_data.quantization_level
            QuantizationUtil.quantize_qwen_models(vae, transformer, bits, weights)
            QwenWeightUtil._set_model_weights(weights, vae, transformer)
            return bits

        raise Exception("Error setting weights")

    @staticmethod
    def _set_model_weights(
        weights: "QwenImageWeightHandler",
        vae: nn.Module,
        transformer: nn.Module,
    ):
        if weights.vae is not None:
            vae.update(weights.vae, strict=False)
        else:
            raise ValueError("No VAE weights loaded from pretrained model")

        if weights.transformer is not None:
            QwenImageTransformerApplier.apply_from_handler(module=transformer, weights=weights.transformer)
        else:
            raise ValueError("No Transformer weights loaded from pretrained model")
