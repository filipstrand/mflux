"""
Weight definitions for the LongCat-Image model.

LongCat-Image uses:
- Flow Match transformer (10 joint + 20 single blocks)
- Qwen2.5-VL text encoder (vision-language model)
- Standard AutoencoderKL VAE (16 latent channels, same as FLUX)
"""

from typing import List

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.tokenizer import LanguageTokenizer
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.flux.weights.flux_weight_mapping import FluxWeightMapping
from mflux.models.longcat.weights.longcat_weight_mapping import LongCatWeightMapping


class LongCatWeightDefinition:
    """
    Weight definitions for the LongCat-Image model.

    Components:
    1. VAE - Standard AutoencoderKL (16 latent channels, same as FLUX)
    2. Transformer - LongCat Flow Match (10 joint + 20 single blocks)
    3. Text Encoder - Qwen2.5-VL (vision-language model)
    """

    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                precision=ModelConfig.precision,
                mapping_getter=FluxWeightMapping.get_vae_mapping,  # Standard VAE
            ),
            ComponentDefinition(
                name="transformer",
                hf_subdir="transformer",
                precision=ModelConfig.precision,
                mapping_getter=LongCatWeightMapping.get_transformer_mapping,
            ),
            ComponentDefinition(
                name="text_encoder",
                hf_subdir="text_encoder",
                model_attr="text_encoder",
                num_blocks=28,  # Qwen2.5-VL has 28 language model layers
                precision=ModelConfig.precision,
                mapping_getter=LongCatWeightMapping.get_text_encoder_mapping,
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
                max_length=512,
                download_patterns=["tokenizer/**"],
            ),
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
        ]

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        """Determine if a module should be quantized."""
        if hasattr(module, "weight") and hasattr(module.weight, "shape"):
            if module.weight.shape[-1] % 64 != 0:
                return False
        return hasattr(module, "to_quantized")
