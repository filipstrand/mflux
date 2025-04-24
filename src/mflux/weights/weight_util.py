from typing import TYPE_CHECKING

import mlx.nn as nn

from mflux.config.config import Config
from mflux.weights.quantization_util import QuantizationUtil

if TYPE_CHECKING:
    from mflux.controlnet.weight_handler_controlnet import WeightHandlerControlnet
    from mflux.flux_tools.redux.weight_handler_redux import WeightHandlerRedux
    from mflux.weights.weight_handler import WeightHandler


class WeightUtil:
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
        weights: "WeightHandler",
        vae: nn.Module,
        transformer: nn.Module,
        t5_text_encoder: nn.Module,
        clip_text_encoder: nn.Module,
    ) -> int | None:
        if weights.meta_data.quantization_level is None and quantize_arg is None:
            WeightUtil._set_model_weights(weights, vae, transformer, t5_text_encoder, clip_text_encoder)
            return None

        if weights.meta_data.quantization_level is None and quantize_arg is not None:
            bits = quantize_arg
            WeightUtil._set_model_weights(weights, vae, transformer, t5_text_encoder, clip_text_encoder)
            QuantizationUtil.quantize_model(vae, transformer, t5_text_encoder, clip_text_encoder, bits, weights)  # fmt:off
            return bits

        if weights.meta_data.quantization_level is not None:
            bits = weights.meta_data.quantization_level
            QuantizationUtil.quantize_model(vae, transformer, t5_text_encoder, clip_text_encoder, bits, weights)  # fmt:off
            WeightUtil._set_model_weights(weights, vae, transformer, t5_text_encoder, clip_text_encoder)
            return bits

        raise Exception("Error setting weights")

    @staticmethod
    def set_controlnet_weights_and_quantize(
        quantize_arg: int | None,
        weights: "WeightHandlerControlnet",
        transformer_controlnet: nn.Module,
    ) -> int | None:
        if weights.meta_data.quantization_level is None and quantize_arg is None:
            transformer_controlnet.update(weights.controlnet_transformer)
            return None

        if weights.meta_data.quantization_level is None and quantize_arg is not None:
            bits = quantize_arg
            transformer_controlnet.update(weights.controlnet_transformer)
            QuantizationUtil.quantize_controlnet(bits, weights, transformer_controlnet)
            return bits

        if weights.meta_data.quantization_level is not None:
            bits = weights.meta_data.quantization_level
            QuantizationUtil.quantize_controlnet(bits, weights, transformer_controlnet)
            transformer_controlnet.update(weights.controlnet_transformer)
            return bits

    @staticmethod
    def _set_model_weights(
        weights: "WeightHandler",
        vae: nn.Module,
        transformer: nn.Module,
        t5_text_encoder: nn.Module,
        clip_text_encoder: nn.Module,
    ):
        vae.update(weights.vae)
        transformer.update(weights.transformer)
        t5_text_encoder.update(weights.t5_encoder)
        clip_text_encoder.update(weights.clip_encoder)

    @staticmethod
    def _set_redux_model_weights(
        weights: "WeightHandlerRedux",
        redux_encoder: nn.Module,
        siglip_vision_transformer: nn.Module,
    ):
        redux_encoder.update(weights.redux_encoder)
        siglip_vision_transformer.update(weights.siglip["vision_model"])

    @staticmethod
    def set_redux_weights_and_quantize(
        quantize_arg: int | None,
        weights: "WeightHandlerRedux",
        redux_encoder: nn.Module,
        siglip_vision_transformer: nn.Module,
    ) -> int | None:
        if weights.meta_data.quantization_level is None and quantize_arg is None:
            WeightUtil._set_redux_model_weights(weights, redux_encoder, siglip_vision_transformer)
            return None

        if weights.meta_data.quantization_level is None and quantize_arg is not None:
            bits = quantize_arg
            WeightUtil._set_redux_model_weights(weights, redux_encoder, siglip_vision_transformer)
            QuantizationUtil.quantize_redux_models(bits, weights, redux_encoder, siglip_vision_transformer)
            return bits

        if weights.meta_data.quantization_level is not None:
            bits = weights.meta_data.quantization_level
            QuantizationUtil.quantize_redux_models(bits, weights, redux_encoder, siglip_vision_transformer)
            WeightUtil._set_redux_model_weights(weights, redux_encoder, siglip_vision_transformer)
            return bits
