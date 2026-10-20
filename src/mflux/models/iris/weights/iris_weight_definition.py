from typing import List

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.tokenizer import LanguageTokenizer
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.flux2.weights.flux2_weight_mapping import Flux2WeightMapping
from mflux.models.iris.weights.iris_weight_mapping import IrisWeightMapping


class IrisWeightDefinition:
    # The mflux layout (what mflux-save writes and model_path reads): transformer/, text_encoder/, tokenizer/.
    # The upstream checkpoints (the Iris repo for the transformer, the Qwen repo for the text encoder and tokenizer)
    # are read through the source_* methods and only when no model_path is given.
    TEXT_ENCODER_REPO = "Qwen/Qwen3-VL-4B-Instruct"
    TEXT_ENCODER_LAYERS = 36
    TEXT_ENCODER_PATTERN = "model-*.safetensors"
    SOURCE_TRANSFORMER_SUBDIR = ""
    SOURCE_TRANSFORMER_PATTERN = "model.safetensors"

    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(name="transformer", hf_subdir="transformer", precision=ModelConfig.precision),
            ComponentDefinition(name="text_encoder", hf_subdir="text_encoder", precision=ModelConfig.precision),
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return [
            TokenizerDefinition(
                name="qwen3_vl",
                hf_subdir="tokenizer",
                tokenizer_class="AutoTokenizer",
                encoder_class=LanguageTokenizer,
                download_patterns=["tokenizer/*"],
            ),
        ]

    @classmethod
    def source_transformer(cls) -> ComponentDefinition:
        return ComponentDefinition(
            name="transformer",
            hf_subdir=cls.SOURCE_TRANSFORMER_SUBDIR,
            precision=ModelConfig.precision,
            mapping_getter=IrisWeightMapping.get_transformer_mapping,
            weight_files=["model.safetensors"],
        )

    @staticmethod
    def source_text_encoder() -> ComponentDefinition:
        # The Qwen3-VL language tower, all 36 layers, from the root of the Qwen repo.
        return ComponentDefinition(
            name="text_encoder",
            hf_subdir="",
            precision=ModelConfig.precision,
            mapping_getter=Flux2WeightMapping.get_text_encoder_mapping,
            num_layers=IrisWeightDefinition.TEXT_ENCODER_LAYERS,
            key_transform=IrisWeightDefinition._text_encoder_key,
        )

    @staticmethod
    def source_tokenizers() -> List[TokenizerDefinition]:
        return [
            TokenizerDefinition(
                name="qwen3_vl",
                hf_subdir="",
                tokenizer_class="AutoTokenizer",
                encoder_class=LanguageTokenizer,
                download_patterns=["tokenizer*", "vocab.json", "merges.txt"],
            ),
        ]

    @staticmethod
    def get_download_patterns() -> List[str]:
        return [
            "transformer/*.safetensors",
            "transformer/*.json",
            "text_encoder/*.safetensors",
            "text_encoder/*.json",
            "tokenizer/*",
        ]

    @staticmethod
    def quantization_predicate(path: str, module, bits: int | None = None) -> bool:
        if not hasattr(module, "to_quantized"):
            return False
        # Tiny pixel-stage layers stay dense; the width check skips SwiGLU w2 (6826) and any other odd input.
        if path.startswith(("pixel_embedder", "final_layer")) or "layer_pool" in path:
            return False
        if path.startswith("pixel_blocks") and ".mlp." in path:
            return False
        # The per-pixel adaLN projections (2560 to 16384) set individual pixel values: at 4 bits and below they
        # print a screen-door grain over flat areas, so they stay dense there (8 bits is clean).
        if path.startswith("pixel_blocks") and path.endswith(".adaln") and bits is not None and bits <= 4:
            return False
        weight = getattr(module, "weight", None)
        return weight is None or weight.shape[-1] % 64 == 0

    @staticmethod
    def _text_encoder_key(key: str) -> str | None:
        prefix = "model.language_model."
        if not key.startswith(prefix):
            return None
        return "model." + key[len(prefix) :]


class IrisTurboWeightDefinition(IrisWeightDefinition):
    SOURCE_TRANSFORMER_SUBDIR = "turbo"
    SOURCE_TRANSFORMER_PATTERN = "turbo/model.safetensors"
