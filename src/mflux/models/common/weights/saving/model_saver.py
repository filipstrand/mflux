from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten
from transformers import PreTrainedTokenizer

from mflux.utils.version_util import VersionUtil


class ModelSaver:
    @staticmethod
    def save_model(
        model: Any,
        bits: int,
        base_path: str,
        tokenizers: list[tuple[str, str]] | None = None,
        components: list[tuple[str, str]] | None = None,
    ) -> None:
        # Default tokenizers: try common patterns
        if tokenizers is None:
            tokenizers = ModelSaver._detect_tokenizers(model)

        # Default components: try common patterns
        if components is None:
            components = ModelSaver._detect_components(model)

        # Save tokenizers
        for attr_path, subdir in tokenizers:
            tokenizer = ModelSaver._get_nested_attr(model, attr_path)
            if tokenizer is not None:
                ModelSaver._save_tokenizer(base_path, tokenizer, subdir)

        # Save model components
        for attr_name, subdir in components:
            component = getattr(model, attr_name, None)
            if component is not None:
                ModelSaver._save_weights(base_path, bits, component, subdir)

    @staticmethod
    def _detect_tokenizers(model: Any) -> list[tuple[str, str]]:
        tokenizers = []
        if hasattr(model, "clip_tokenizer") and hasattr(model.clip_tokenizer, "tokenizer"):
            tokenizers.append(("clip_tokenizer.tokenizer", "tokenizer"))
        if hasattr(model, "t5_tokenizer") and hasattr(model.t5_tokenizer, "tokenizer"):
            tokenizers.append(("t5_tokenizer.tokenizer", "tokenizer_2"))
        if hasattr(model, "qwen_tokenizer") and hasattr(model.qwen_tokenizer, "tokenizer"):
            tokenizers.append(("qwen_tokenizer.tokenizer", "tokenizer"))
        if hasattr(model, "fibo_tokenizer") and hasattr(model.fibo_tokenizer, "tokenizer"):
            tokenizers.append(("fibo_tokenizer.tokenizer", "tokenizer"))
        return tokenizers

    @staticmethod
    def _detect_components(model: Any) -> list[tuple[str, str]]:
        components = []
        if hasattr(model, "vae"):
            components.append(("vae", "vae"))
        if hasattr(model, "transformer"):
            components.append(("transformer", "transformer"))
        if hasattr(model, "clip_text_encoder"):
            components.append(("clip_text_encoder", "text_encoder"))
        if hasattr(model, "t5_text_encoder"):
            components.append(("t5_text_encoder", "text_encoder_2"))
        if hasattr(model, "text_encoder"):
            # Only add if we haven't already added clip_text_encoder or t5_text_encoder
            if not any(c[1] == "text_encoder" for c in components):
                components.append(("text_encoder", "text_encoder"))
        return components

    @staticmethod
    def _get_nested_attr(obj: Any, attr_path: str) -> Any:
        attrs = attr_path.split(".")
        result = obj
        for attr in attrs:
            if not hasattr(result, attr):
                return None
            result = getattr(result, attr)
        return result

    @staticmethod
    def _save_tokenizer(base_path: str, tokenizer: PreTrainedTokenizer, subdir: str) -> None:
        path = Path(base_path) / subdir
        path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(path)

    @staticmethod
    def _save_weights(base_path: str, bits: int, model: nn.Module, subdir: str) -> None:
        path = Path(base_path) / subdir
        path.mkdir(parents=True, exist_ok=True)
        weights = ModelSaver._split_weights(base_path, dict(tree_flatten(model.parameters())))
        for i, weight in enumerate(weights):
            mx.save_safetensors(
                str(path / f"{i}.safetensors"),
                weight,
                {
                    "quantization_level": str(bits),
                    "mflux_version": VersionUtil.get_mflux_version(),
                },
            )

    @staticmethod
    def _split_weights(base_path: str, weights: dict, max_file_size_gb: int = 2) -> list:
        max_file_size_bytes = max_file_size_gb << 30
        shards = []
        shard, shard_size = {}, 0
        for k, v in weights.items():
            if shard_size + v.nbytes > max_file_size_bytes:
                shards.append(shard)
                shard, shard_size = {}, 0
            shard[k] = v
            shard_size += v.nbytes
        shards.append(shard)
        return shards
