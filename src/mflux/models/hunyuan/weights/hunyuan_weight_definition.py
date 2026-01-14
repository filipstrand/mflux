"""Weight definitions for Hunyuan-DiT model.

Defines which components to load and from where.
"""

from typing import List

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.tokenizer import LanguageTokenizer
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.hunyuan.weights.hunyuan_weight_mapping import HunyuanWeightMapping


class HunyuanWeightDefinition:
    """
    Weight definitions for Hunyuan-DiT model.

    Components:
    - transformer: HunyuanDiT2DModel (28 blocks)
    - vae: Standard 4-channel VAE
    - clip_encoder: Chinese CLIP (1024 dim)
    - t5_encoder: mT5-XXL (2048 dim)
    """

    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                precision=ModelConfig.precision,
                mapping_getter=HunyuanWeightMapping.get_vae_mapping,
            ),
            ComponentDefinition(
                name="transformer",
                hf_subdir="transformer",
                precision=ModelConfig.precision,
                mapping_getter=HunyuanWeightMapping.get_transformer_mapping,
            ),
            ComponentDefinition(
                name="clip_encoder",
                hf_subdir="text_encoder",
                model_attr="clip_text_encoder",
                precision=ModelConfig.precision,
                mapping_getter=HunyuanWeightMapping.get_clip_encoder_mapping,
            ),
            ComponentDefinition(
                name="t5_encoder",
                hf_subdir="text_encoder_2",
                model_attr="t5_text_encoder",
                num_blocks=24,
                precision=ModelConfig.precision,
                mapping_getter=HunyuanWeightMapping.get_t5_encoder_mapping,
            ),
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return [
            TokenizerDefinition(
                name="clip",
                hf_subdir="tokenizer",
                tokenizer_class="CLIPTokenizer",
                encoder_class=LanguageTokenizer,
                max_length=77,
                download_patterns=["tokenizer/**"],
            ),
            TokenizerDefinition(
                name="t5",
                hf_subdir="tokenizer_2",
                tokenizer_class="T5Tokenizer",
                encoder_class=LanguageTokenizer,
                max_length=256,
                download_patterns=["tokenizer_2/**"],
            ),
        ]

    @staticmethod
    def get_download_patterns() -> List[str]:
        return [
            "text_encoder/*.safetensors",
            "text_encoder/*.json",
            "text_encoder_2/*.safetensors",
            "text_encoder_2/*.json",
            "transformer/*.safetensors",
            "transformer/*.json",
            "vae/*.safetensors",
            "vae/*.json",
        ]

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        """
        Determine if a module should be quantized.

        Quantize Linear layers with compatible dimensions (divisible by 64).
        """
        if hasattr(module, "weight") and hasattr(module.weight, "shape"):
            # Skip layers with dimensions not divisible by 64
            if module.weight.shape[-1] % 64 != 0:
                return False
        return hasattr(module, "to_quantized")
