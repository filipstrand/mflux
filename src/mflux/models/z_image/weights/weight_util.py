from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

from mflux.models.common.config import ModelConfig

if TYPE_CHECKING:
    from mflux.models.z_image.weights.weight_handler import WeightHandler


class WeightUtil:
    @staticmethod
    def set_weights_and_quantize(
        quantize_arg: int | None,
        weights: "WeightHandler",
        vae: nn.Module,
        transformer: nn.Module | None = None,
        text_encoder: nn.Module | None = None,
    ) -> int | None:
        # Convert weights to target precision (bfloat16) for performance
        WeightUtil._convert_weights_precision(weights)

        # No quantization
        if weights.meta_data.quantization_level is None and quantize_arg is None:
            WeightUtil._set_model_weights(weights, vae, transformer, text_encoder)
            return None

        # Quantize non-quantized weights
        if weights.meta_data.quantization_level is None and quantize_arg is not None:
            bits = quantize_arg
            WeightUtil._set_model_weights(weights, vae, transformer, text_encoder)
            WeightUtil._quantize_model(vae, transformer, text_encoder, bits)
            return bits

        # Weights are already quantized
        if weights.meta_data.quantization_level is not None:
            bits = weights.meta_data.quantization_level
            WeightUtil._quantize_model(vae, transformer, text_encoder, bits)
            WeightUtil._set_model_weights(weights, vae, transformer, text_encoder)
            return bits

        raise Exception("Error setting weights")

    @staticmethod
    def _convert_weights_precision(weights: "WeightHandler") -> None:
        def convert_dict(d: dict) -> dict:
            for key, value in d.items():
                if isinstance(value, mx.array):
                    d[key] = value.astype(ModelConfig.precision)
                elif isinstance(value, dict):
                    convert_dict(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            convert_dict(item)
            return d

        if weights.vae:
            convert_dict(weights.vae)
        if weights.transformer:
            convert_dict(weights.transformer)
        if weights.text_encoder:
            convert_dict(weights.text_encoder)

    @staticmethod
    def _set_model_weights(
        weights: "WeightHandler",
        vae: nn.Module,
        transformer: nn.Module | None = None,
        text_encoder: nn.Module | None = None,
    ) -> None:
        vae.update(weights.vae, strict=False)
        if transformer is not None and weights.transformer is not None:
            transformer.update(weights.transformer, strict=False)
        if text_encoder is not None and weights.text_encoder is not None:
            text_encoder.update(weights.text_encoder, strict=False)

    @staticmethod
    def _quantize_model(
        vae: nn.Module,
        transformer: nn.Module | None,
        text_encoder: nn.Module | None,
        bits: int,
    ) -> None:
        nn.quantize(vae, bits=bits)
        if transformer is not None:
            nn.quantize(transformer, bits=bits)
        if text_encoder is not None:
            nn.quantize(text_encoder, bits=bits)
