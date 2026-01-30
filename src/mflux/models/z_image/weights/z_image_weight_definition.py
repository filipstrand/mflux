from typing import List

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.tokenizer import LanguageTokenizer
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.z_image.weights.z_image_weight_mapping import ZImageWeightMapping

# Alignment required for efficient INT4/INT8 quantization (matches MLX group size)
QUANTIZATION_ALIGNMENT = 64

# Attention projection layers that should be kept at full precision for quality
# These layers are most sensitive to quantization error
ATTENTION_SENSITIVE_LAYERS = frozenset(
    [
        "to_q",
        "to_k",
        "to_v",  # Z-Image attention naming
        "q_proj",
        "k_proj",
        "v_proj",  # Common transformer naming
    ]
)


class ZImageWeightDefinition:
    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                num_blocks=4,
                precision=ModelConfig.precision,
                mapping_getter=ZImageWeightMapping.get_vae_mapping,
            ),
            ComponentDefinition(
                name="transformer",
                hf_subdir="transformer",
                num_layers=30,
                precision=ModelConfig.precision,
                mapping_getter=ZImageWeightMapping.get_transformer_mapping,
            ),
            ComponentDefinition(
                name="text_encoder",
                hf_subdir="text_encoder",
                num_layers=36,
                precision=ModelConfig.precision,
                mapping_getter=ZImageWeightMapping.get_text_encoder_mapping,
            ),
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return [
            TokenizerDefinition(
                name="z_image",
                hf_subdir="tokenizer",
                tokenizer_class="AutoTokenizer",
                encoder_class=LanguageTokenizer,
                max_length=512,
                use_chat_template=True,
                chat_template_kwargs={"enable_thinking": True},
                download_patterns=["tokenizer/*"],
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
            "tokenizer/*",
        ]

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        """Determine if a module should be quantized.

        Improved predicate that:
        - Skips modules that can't be quantized
        - Preserves attention QKV at higher precision for quality
        - Skips VAE components to prevent color banding
        - Checks dimension alignment (misaligned dims cause quality issues)
        """
        if not hasattr(module, "to_quantized"):
            return False

        # Skip VAE entirely - quantizing VAE causes color banding artifacts
        if "vae" in path:
            return False

        # Keep attention QKV projections at higher precision for quality
        # Use set membership for O(1) lookup
        if any(layer in path for layer in ATTENTION_SENSITIVE_LAYERS):
            return False

        # Check dimension alignment - misaligned weights cause quality issues
        # Explicit None check to handle modules where weight exists but is None
        if hasattr(module, "weight") and module.weight is not None and hasattr(module.weight, "shape"):
            if module.weight.shape[-1] % QUANTIZATION_ALIGNMENT != 0:
                return False

        return True
