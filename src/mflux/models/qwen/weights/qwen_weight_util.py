
import mlx.nn as nn

from mflux.config.config import Config
from mflux.models.qwen.model.qwen_transformer.qwen_transformer import QwenImageTransformerApplier
from mflux.models.qwen.weights.qwen_text_encoder_loader import QwenTextEncoderLoader
from mflux.models.qwen.weights.qwen_weight_handler import QwenImageWeightHandler
from mflux.utils.quantization_util import QuantizationUtil


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
        weights: QwenImageWeightHandler,
        vae: nn.Module,
        transformer: nn.Module,
        text_encoder: nn.Module | None = None,
    ):
        vae.update(weights.vae, strict=False)
        QwenImageTransformerApplier.apply_from_handler(module=transformer, weights=weights.transformer)
        nested_weights = QwenTextEncoderLoader.convert_to_nested_dict(weights.qwen_text_encoder)
        text_encoder.update(nested_weights, strict=False)
