from typing import List

from mflux.models.common.tokenizer import VisionLanguageTokenizer
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.fibo_vlm.tokenizer.qwen2vl_processor import Qwen2VLProcessor
from mflux.models.fibo_vlm.weights.fibo_vlm_weight_mapping import FIBOVLMWeightMapping


class FIBOVLMWeightDefinition:
    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="decoder",
                hf_subdir="",
                num_blocks=36,
                loading_mode="torch_bfloat16",
                weight_prefix_filters=["model.language_model", "lm_head"],
                mapping_getter=lambda: FIBOVLMWeightMapping.get_vlm_decoder_mapping(36),
            ),
            ComponentDefinition(
                name="visual",
                hf_subdir="",
                num_blocks=24,
                loading_mode="torch_bfloat16",
                weight_prefix_filters=["model.visual"],
                mapping_getter=lambda: FIBOVLMWeightMapping.get_vlm_visual_mapping(24),
            ),
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return [
            TokenizerDefinition(
                name="fibo_vlm",
                hf_subdir=".",  # Root directory
                tokenizer_class="Qwen2Tokenizer",
                encoder_class=VisionLanguageTokenizer,
                processor_class=Qwen2VLProcessor,
                max_length=1024,
                fallback_subdirs=["tokenizer", "text_encoder"],
                download_patterns=["vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json"],
            ),
        ]

    @staticmethod
    def get_download_patterns() -> List[str]:
        return [
            "*.safetensors",
            "*.json",
            "vocab.json",
            "merges.txt",
            "tokenizer.json",
            "tokenizer_config.json",
        ]

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        # Skip lm_head - it maps to vocab size and quantization corrupts dimensions
        if "lm_head" in path:
            return False
        return hasattr(module, "to_quantized")
