"""Weight definitions for NewBie-image model.

Defines which components to load and from where.
"""

from typing import List

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.tokenizer import LanguageTokenizer
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.newbie.weights.newbie_weight_mapping import NewBieWeightMapping


class NewBieWeightDefinition:
    """
    Weight definitions for NewBie-image model.

    Components:
    - transformer: NextDiT (36 blocks, GQA)
    - vae: FLUX.1-dev VAE (16 channels)
    - gemma3_encoder: Gemma3-4B-it (2560 dim)
    - jina_clip_encoder: Jina CLIP v2 (1024 dim)
    """

    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                precision=ModelConfig.precision,
                mapping_getter=NewBieWeightMapping.get_vae_mapping,
            ),
            ComponentDefinition(
                name="transformer",
                hf_subdir="transformer",
                precision=ModelConfig.precision,
                mapping_getter=NewBieWeightMapping.get_transformer_mapping,
            ),
            ComponentDefinition(
                name="gemma3_encoder",
                hf_subdir="text_encoder",
                model_attr="gemma3_text_encoder",
                num_blocks=36,
                precision=ModelConfig.precision,
                mapping_getter=NewBieWeightMapping.get_gemma3_encoder_mapping,
            ),
            ComponentDefinition(
                name="jina_clip_encoder",
                hf_subdir="text_encoder_2",
                model_attr="jina_clip_encoder",
                num_blocks=24,
                precision=ModelConfig.precision,
                mapping_getter=NewBieWeightMapping.get_jina_clip_encoder_mapping,
            ),
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return [
            TokenizerDefinition(
                name="gemma3",
                hf_subdir="tokenizer",
                tokenizer_class="GemmaTokenizer",
                encoder_class=LanguageTokenizer,
                max_length=512,
                download_patterns=["tokenizer/**"],
            ),
            TokenizerDefinition(
                name="jina_clip",
                hf_subdir="tokenizer_2",
                tokenizer_class="BertTokenizer",
                encoder_class=LanguageTokenizer,
                max_length=77,
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
            "scheduler/**",
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
