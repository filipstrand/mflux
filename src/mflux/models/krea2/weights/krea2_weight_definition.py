"""Weight definition for Krea-2.

Components: the Qwen-Image VAE (reused from the ``qwen`` family) and the custom
Krea-2 single-stream DiT (28 layers).

Open items (see NOTES.md):
  * Text encoder: Qwen3-VL-4B (12-layer tap) is not yet wired as a component.
  * Weight source layout: ``Lumatrix/Krea-2`` ships single-file ``turbo``/``raw``
    safetensors at the repo root, not in a ``transformer/`` subdir. This
    definition assumes the mflux-standard assembled layout (the eventual
    ``+save`` output); point ``model_path`` at a local assembled dir for dev.
"""

from typing import List

import mlx.core as mx

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.tokenizer import LanguageTokenizer
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.krea2.model.krea2_text_encoder.text_encoder import KREA2_TEMPLATE
from mflux.models.krea2.weights.krea2_weight_mapping import Krea2WeightMapping

# The official Krea-2-Turbo TE uses `language_model.*`; the standalone
# Qwen/Qwen3-VL-4B-Instruct repo uses `model.language_model.*`. Accept both.
_TE_PREFIXES = ("model.language_model.", "language_model.")


def _strip_te_prefix(key: str) -> str | None:
    # Keep only the Qwen3-VL language model and strip its prefix so keys match the
    # flat Krea2TextEncoder module (embed_tokens / layers.N / norm). Drops vision keys.
    for prefix in _TE_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return None


class Krea2WeightDefinition:
    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                loading_mode="single",
                mapping_getter=Krea2WeightMapping.get_vae_mapping,
            ),
            ComponentDefinition(
                name="transformer",
                hf_subdir="transformer",
                loading_mode="multi_glob",
                precision=ModelConfig.precision,
                mapping_getter=Krea2WeightMapping.get_transformer_mapping,
                num_layers=28,
            ),
            ComponentDefinition(
                name="text_encoder",
                hf_subdir="text_encoder",
                loading_mode="mlx_native",
                precision=mx.bfloat16,
                skip_quantization=True,  # quantizing the TE degrades conditioning
                mapping_getter=None,  # direct load; key_transform strips prefix to match module paths
                key_transform=_strip_te_prefix,
            ),
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return [
            TokenizerDefinition(
                name="qwen3vl",
                hf_subdir="tokenizer",
                tokenizer_class="AutoTokenizer",
                encoder_class=LanguageTokenizer,
                max_length=1024,
                padding="longest",
                template=KREA2_TEMPLATE,
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
            "tokenizer/**",
        ]

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        # Skip layers whose input dim isn't divisible by the group size (e.g. the
        # txtfusion projector, Linear(12->1)) — mx.quantize requires last-dim % 64 == 0.
        if not hasattr(module, "to_quantized"):
            return False
        weight = getattr(module, "weight", None)
        return weight is not None and weight.shape[-1] % 64 == 0
