"""Weight definitions for FLUX.2 model.

Defines which components to load and how they map to model attributes.
"""

from typing import List

import mlx.nn as nn

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.tokenizer import LanguageTokenizer
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.flux2.weights.flux2_weight_mapping import Flux2WeightMapping


class Flux2WeightDefinition:
    """Weight definitions for FLUX.2 model."""

    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        """Get component definitions for FLUX.2."""
        return [
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                precision=ModelConfig.precision,
                mapping_getter=Flux2WeightMapping.get_vae_mapping,
            ),
            ComponentDefinition(
                name="transformer",
                hf_subdir="transformer",
                precision=ModelConfig.precision,
                mapping_getter=Flux2WeightMapping.get_transformer_mapping,
            ),
            ComponentDefinition(
                name="text_encoder",
                hf_subdir="text_encoder",
                model_attr="text_encoder",
                num_blocks=40,  # Mistral3 has 40 layers
                precision=ModelConfig.precision,
                mapping_getter=Flux2WeightMapping.get_text_encoder_mapping,
            ),
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        """Get tokenizer definitions for FLUX.2."""
        return [
            TokenizerDefinition(
                name="mistral3",
                hf_subdir="tokenizer",
                tokenizer_class="PreTrainedTokenizerFast",
                encoder_class=LanguageTokenizer,
                max_length=1024,  # Default max length
                download_patterns=["tokenizer/**"],
            ),
        ]

    @staticmethod
    def get_download_patterns() -> List[str]:
        """Get download patterns for FLUX.2 model files."""
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
        """Determine if a module should be quantized.

        Args:
            path: Module path in the model
            module: The module instance

        Returns:
            True if the module should be quantized
        """
        return hasattr(module, "to_quantized")
