from dataclasses import replace
from typing import List

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.tokenizer import LanguageTokenizer
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.flux2.weights.flux2_weight_mapping import Flux2WeightMapping
from mflux.models.flux3.weights.flux3_weight_mapping import Flux3WeightMapping


class Flux3KleinWeightDefinition:
    # The text encoder is the Qwen3-VL-4B language tower; layers beyond the last output layer (32) are never read.
    TEXT_ENCODER_LAYERS = 32

    @staticmethod
    def _text_encoder_key(key: str) -> str | None:
        prefix = "model.language_model."
        if not key.startswith(prefix):
            return None  # vision tower
        name = key[len(prefix) :]
        if name.startswith("layers.") and int(name.split(".")[1]) >= Flux3KleinWeightDefinition.TEXT_ENCODER_LAYERS:
            return None
        return "model." + name

    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                precision=ModelConfig.precision,
                mapping_getter=Flux3WeightMapping.get_vae_mapping,
            ),
            ComponentDefinition(
                name="transformer",
                hf_subdir="transformer",
                precision=ModelConfig.precision,
                mapping_getter=Flux3WeightMapping.get_transformer_mapping,
            ),
            ComponentDefinition(
                name="text_encoder",
                hf_subdir="text_encoder",
                precision=ModelConfig.precision,
                mapping_getter=Flux2WeightMapping.get_text_encoder_mapping,
                num_layers=Flux3KleinWeightDefinition.TEXT_ENCODER_LAYERS,
                key_transform=Flux3KleinWeightDefinition._text_encoder_key,
            ),
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return [
            TokenizerDefinition(
                name="qwen3_vl",
                hf_subdir="text_encoder",
                tokenizer_class="AutoTokenizer",
                encoder_class=LanguageTokenizer,
                max_length=4096,
                use_chat_template=True,
                download_patterns=[
                    "text_encoder/tokenizer*",
                    "text_encoder/vocab.json",
                    "text_encoder/merges.txt",
                    "text_encoder/chat_template.json",
                    "text_encoder/chat_template.jinja",
                ],  # fmt: skip
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
            "text_encoder/merges.txt",
            "text_encoder/chat_template.jinja",
        ]

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        return hasattr(module, "to_quantized")


class Flux3KleinBaseWeightDefinition(Flux3KleinWeightDefinition):
    # Klein Base is undistilled: same layout as the distilled checkpoint minus target_timestep_embedder.
    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            replace(component, mapping_getter=Flux3WeightMapping.get_base_transformer_mapping)
            if component.name == "transformer"
            else component
            for component in Flux3KleinWeightDefinition.get_components()
        ]
