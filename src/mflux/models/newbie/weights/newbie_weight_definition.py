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
            # Note: Jina CLIP encoder is loaded separately from jinaai/jina-clip-v2
            # The projection layers (clip_text_pooled_proj) are in the transformer weights
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return [
            TokenizerDefinition(
                name="gemma3",
                hf_subdir="text_encoder",
                tokenizer_class="GemmaTokenizer",
                encoder_class=LanguageTokenizer,
                max_length=512,
                download_patterns=["text_encoder/*.json"],
            ),
            # Note: Jina CLIP tokenizer is loaded from jinaai/jina-clip-v2 repo
        ]

    @staticmethod
    def get_download_patterns() -> List[str]:
        return [
            "text_encoder/*.safetensors",
            "text_encoder/*.json",
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
