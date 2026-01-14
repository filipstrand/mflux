from typing import List

from mflux.models.chroma.weights.chroma_weight_mapping import ChromaWeightMapping
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.tokenizer import LanguageTokenizer
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.flux.weights.flux_weight_mapping import FluxWeightMapping


class ChromaWeightDefinition:
    """
    Weight definitions for the Chroma model.

    Key differences from FLUX:
    1. T5 encoder is at 'text_encoder/' (not 'text_encoder_2/')
    2. Tokenizer is at 'tokenizer/' (not 'tokenizer_2/')
    3. No CLIP encoder (T5-only text encoding)
    4. Uses ChromaWeightMapping for transformer (DistilledGuidanceLayer)
    """

    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                precision=ModelConfig.precision,
                mapping_getter=FluxWeightMapping.get_vae_mapping,  # Reuse FLUX VAE
            ),
            ComponentDefinition(
                name="transformer",
                hf_subdir="transformer",
                precision=ModelConfig.precision,
                mapping_getter=ChromaWeightMapping.get_transformer_mapping,
            ),
            ComponentDefinition(
                name="t5_encoder",
                hf_subdir="text_encoder",  # Chroma uses 'text_encoder' not 'text_encoder_2'
                model_attr="t5_text_encoder",
                num_blocks=24,
                precision=ModelConfig.precision,
                mapping_getter=FluxWeightMapping.get_t5_encoder_mapping,  # Reuse FLUX T5 mapping
            ),
            # No CLIP encoder - Chroma uses T5-only
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return [
            TokenizerDefinition(
                name="t5",
                hf_subdir="tokenizer",  # Chroma uses 'tokenizer' not 'tokenizer_2'
                tokenizer_class="T5Tokenizer",
                encoder_class=LanguageTokenizer,
                max_length=512,  # Will be overridden by model_config.max_sequence_length
                download_patterns=["tokenizer/**"],
            ),
            # No CLIP tokenizer - Chroma uses T5-only
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
        """
        Determine if a module should be quantized.

        Same logic as FLUX - quantize Linear layers with compatible dimensions.
        """
        if hasattr(module, "weight") and hasattr(module.weight, "shape"):
            # Skip layers with dimensions not divisible by 64
            if module.weight.shape[-1] % 64 != 0:
                return False
        return hasattr(module, "to_quantized")
