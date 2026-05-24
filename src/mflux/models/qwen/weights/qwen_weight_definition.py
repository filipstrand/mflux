from typing import List

import mlx.core as mx

from mflux.models.common.tokenizer import LanguageTokenizer
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.qwen.weights.qwen_weight_mapping import QwenWeightMapping


class QwenWeightDefinition:
    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                loading_mode="single",
                mapping_getter=QwenWeightMapping.get_vae_mapping,
            ),
            ComponentDefinition(
                name="transformer",
                hf_subdir="transformer",
                loading_mode="multi_glob",
                mapping_getter=QwenWeightMapping.get_transformer_mapping,
            ),
            ComponentDefinition(
                name="text_encoder",
                hf_subdir="text_encoder",
                loading_mode="multi_json",
                precision=mx.bfloat16,
                skip_quantization=True,  # Quantization causes significant semantic degradation
                mapping_getter=QwenWeightMapping.get_text_encoder_mapping,
            ),
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return [
            TokenizerDefinition(
                name="qwen",
                hf_subdir="tokenizer",
                tokenizer_class="Qwen2Tokenizer",
                encoder_class=LanguageTokenizer,
                max_length=1058,
                padding="longest",
                template="<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                download_patterns=["tokenizer/**", "added_tokens.json", "chat_template.jinja"],
            ),
        ]

    @staticmethod
    def get_download_patterns() -> List[str]:
        return [
            "vae/*.safetensors",
            "vae/*.json",
            "transformer/*.safetensors",
            "transformer/*.json",
            "text_encoder/*.safetensors",
            "text_encoder/*.json",
        ]

    @staticmethod
    def quantization_predicate(path: str, module, bits: int | None = None) -> bool:
        if not hasattr(module, "to_quantized"):
            return False

        if bits == 4 and QwenWeightDefinition._is_q4_sensitive_transformer_path(path):
            return False

        return True

    @staticmethod
    def quantization_predicate_for_loaded_weights(weights: LoadedWeights | None, bits: int | None):
        if bits == 4 and QwenWeightDefinition._should_use_mixed_q4(weights):
            return QwenWeightDefinition.quantization_predicate

        return QwenWeightDefinition._quantize_all_predicate

    @staticmethod
    def _is_q4_sensitive_transformer_path(path: str) -> bool:
        if path in {"img_in", "txt_in", "norm_out.linear", "proj_out"}:
            return True

        if path.startswith("time_text_embed."):
            return True

        return ".img_mod_linear" in path or ".txt_mod_linear" in path

    @staticmethod
    def _should_use_mixed_q4(weights: LoadedWeights | None) -> bool:
        if weights is None:
            return True

        transformer = weights.components.get("transformer")
        if not isinstance(transformer, dict):
            return True

        # Old saved q4 Qwen models quantized every Linear, so img_in has quantized
        # scales. New mixed q4 keeps img_in as a regular Linear and has no scales.
        return not QwenWeightDefinition._has_nested_key(transformer, "img_in.scales")

    @staticmethod
    def _has_nested_key(weights: dict, path: str) -> bool:
        current = weights
        for part in path.split("."):
            if isinstance(current, list):
                if not part.isdigit():
                    return False
                index = int(part)
                if index >= len(current):
                    return False
                current = current[index]
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return False
        return True

    @staticmethod
    def _quantize_all_predicate(path: str, module, bits: int | None = None) -> bool:
        return hasattr(module, "to_quantized")
