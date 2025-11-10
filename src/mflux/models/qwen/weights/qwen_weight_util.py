from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten  # noqa: F401

from mflux.config.config import Config
from mflux.models.qwen.model.qwen_text_encoder.qwen_vision_transformer import VisionTransformer
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
            # Conv3d weights: from (O,I,D,H,W) to (O,D,H,W,I) for MLX Conv3d
            # Original: (out_channels, in_channels, temporal, height, width)
            # MLX wants: (out_channels, temporal, height, width, in_channels)
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
    def _convert_weights_to_bf16(weights_dict: dict):
        """
        Recursively convert all weight tensors to BF16.
        This matches PyTorch's behavior where the text encoder model is loaded in BF16.
        """

        def convert_recursive(obj):
            if isinstance(obj, mx.array):
                return obj.astype(mx.bfloat16)
            elif isinstance(obj, dict):
                return {k: convert_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_recursive(item) for item in obj]
            else:
                return obj

        return convert_recursive(weights_dict)

    @staticmethod
    def _set_model_weights(
        weights: "QwenWeightHandler",
        vae: nn.Module,
        transformer: nn.Module,
        text_encoder: nn.Module | None = None,
    ):
        vae.update(weights.vae, strict=False)
        transformer.update(weights.transformer, strict=False)

        # Check if visual weights are present and create visual transformer if needed
        if text_encoder is not None:
            has_visual_weights = (
                "encoder" in weights.qwen_text_encoder and "visual" in weights.qwen_text_encoder["encoder"]
            )
            if has_visual_weights and text_encoder.encoder.visual is None:
                print("🔧 Initializing VisionTransformer for Edit model")
                text_encoder.encoder.visual = VisionTransformer(
                    patch_size=14,
                    temporal_patch_size=2,
                    in_channels=3,
                    embed_dim=1280,
                    depth=32,
                    num_heads=16,
                    mlp_ratio=2.671875,
                    hidden_size=text_encoder.encoder.hidden_size,
                    spatial_merge_size=2,  # Match HF's spatial merging
                )

            # Convert all text encoder weights to BF16 to match PyTorch's behavior
            # PyTorch loads the model in BF16 and performs all computations in BF16
            # This ensures numerical consistency between MLX and PyTorch
            weights.qwen_text_encoder = QwenWeightUtil._convert_weights_to_bf16(weights.qwen_text_encoder)

            text_encoder.update(weights.qwen_text_encoder, strict=False)
