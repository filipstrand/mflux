from typing import List

from mflux.models.common.tokenizer import VisionLanguageTokenizer
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.fibo_vlm.tokenizer.qwen2vl_processor import Qwen2VLProcessor
from mflux.models.fibo_vlm.weights.fibo_vlm_weight_mapping import FIBOVLMWeightMapping

# Qwen2VL chat template (required for apply_chat_template)
QWEN2VL_CHAT_TEMPLATE = """{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system
You are a helpful assistant.<|im_end|>
{% endif %}<|im_start|>{{ message['role'] }}
{% if message['content'] is string %}{{ message['content'] }}<|im_end|>
{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>
{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""  # noqa: E501


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
                chat_template=QWEN2VL_CHAT_TEMPLATE,
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
