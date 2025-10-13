from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten
from transformers import Qwen2Tokenizer

from mflux.utils.version_util import VersionUtil


class QwenModelSaver:
    @staticmethod
    def save_model(model, bits: int, base_path: str):
        # Save the tokenizer
        QwenModelSaver._save_tokenizer(base_path, model.qwen_tokenizer.tokenizer, "tokenizer")

        # Save the models
        QwenModelSaver.save_weights(base_path, bits, model.vae, "vae")
        QwenModelSaver.save_weights(base_path, bits, model.transformer, "transformer")
        QwenModelSaver.save_weights(base_path, bits, model.text_encoder, "text_encoder")

    @staticmethod
    def _save_tokenizer(base_path: str, tokenizer: Qwen2Tokenizer, subdir: str):
        path = Path(base_path) / subdir
        tokenizer.save_pretrained(path)

    @staticmethod
    def save_weights(base_path: str, bits: int, model: nn.Module, subdir: str):
        path = Path(base_path) / subdir
        path.mkdir(parents=True, exist_ok=True)
        weights = QwenModelSaver._split_weights(base_path, dict(tree_flatten(model.parameters())))
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
        # Copied from mlx-examples repo
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

