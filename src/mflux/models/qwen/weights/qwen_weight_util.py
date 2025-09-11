from typing import TYPE_CHECKING

import mlx.nn as nn
from mlx.utils import tree_flatten  # noqa: F401

from mflux.config.config import Config
from mflux.utils.quantization_util import QuantizationUtil

if TYPE_CHECKING:
    from mflux.models.qwen.weights.qwen_weight_handler import QwenWeightHandler


class QwenWeightUtil:
    @staticmethod
    def flatten(params):
        return [(k, v) for p in params for (k, v) in p]

    @staticmethod
    def reshape_weights(key, value):
        if len(value.shape) == 4:
            value = value.transpose(0, 2, 3, 1)
        elif len(value.shape) == 5:
            value = value.transpose(0, 2, 3, 4, 1)
        value = value.reshape(-1).reshape(value.shape).astype(Config.precision)
        return [(key, value)]

    @staticmethod
    def set_weights_and_quantize(
        quantize_arg: int | None,
        weights: "QwenWeightHandler",
        vae: nn.Module,
        transformer: nn.Module,
        text_encoder: nn.Module | None = None,
    ) -> int | None:
        if weights.meta_data.quantization_level is None and quantize_arg is None:
            QwenWeightUtil._set_model_weights(weights, vae, transformer, text_encoder)
            return None

        if weights.meta_data.quantization_level is None and quantize_arg is not None:
            bits = quantize_arg
            QwenWeightUtil._set_model_weights(weights, vae, transformer, text_encoder)
            QuantizationUtil.quantize_qwen_models(text_encoder, vae, transformer, bits, weights)
            return bits

        if weights.meta_data.quantization_level is not None:
            bits = weights.meta_data.quantization_level
            QuantizationUtil.quantize_qwen_models(text_encoder, vae, transformer, bits, weights)
            QwenWeightUtil._set_model_weights(weights, vae, transformer, text_encoder)
            return bits

        raise Exception("Error setting weights")

    @staticmethod
    def _set_model_weights(
        weights: "QwenWeightHandler",
        vae: nn.Module,
        transformer: nn.Module,
        text_encoder: nn.Module | None = None,
    ):
        vae.update(weights.vae, strict=False)
        transformer.update(weights.transformer, strict=False)
        text_encoder.update(weights.qwen_text_encoder, strict=False)
