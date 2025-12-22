from typing import List

import mlx.core as mx

from mflux.models.common.tokenizer import LanguageTokenizer
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.qwen_layered.weights.qwen_layered_weight_mapping import QwenLayeredWeightMapping


class QwenLayeredWeightDefinition:
    """
    Weight definition for Qwen-Image-Layered model.
    
    Components:
    - VAE: RGBA-VAE with 4-channel I/O
    - Transformer: VLD-MMDiT with Layer3D RoPE
    - Text Encoder: Qwen2.5-VL (same as base, skip quantization)
    """

    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                loading_mode="single",
                mapping_getter=QwenLayeredWeightMapping.get_vae_mapping,
            ),
            ComponentDefinition(
                name="transformer",
                hf_subdir="transformer",
                loading_mode="multi_glob",
                mapping_getter=QwenLayeredWeightMapping.get_transformer_mapping,
            ),
            ComponentDefinition(
                name="text_encoder",
                hf_subdir="text_encoder",
                loading_mode="multi_json",
                precision=mx.bfloat16,
                skip_quantization=True,  # Quantization causes significant semantic degradation
                mapping_getter=QwenLayeredWeightMapping.get_text_encoder_mapping,
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
                max_length=1024,
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
    def quantization_predicate(path: str, module) -> bool:
        """
        Determine if a module should be quantized.
        
        Quantizes all modules with to_quantized method (nn.Linear, etc.)
        except those in skip_quantization components (text_encoder).
        """
        return hasattr(module, "to_quantized")
