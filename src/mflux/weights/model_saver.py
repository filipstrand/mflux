from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten
from transformers import CLIPTokenizer, T5Tokenizer


class ModelSaver:
    @staticmethod
    def save_model(model, bits: int, base_path: str):
        # Save the tokenizers
        ModelSaver._save_tokenizer(base_path, model.clip_tokenizer.tokenizer, "tokenizer")
        ModelSaver._save_tokenizer(base_path, model.t5_tokenizer.tokenizer, "tokenizer_2")

        # Save the models
        ModelSaver.save_weights(base_path, bits, model.vae, "vae")
        ModelSaver.save_weights(base_path, bits, model.transformer, "transformer")
        ModelSaver.save_weights(base_path, bits, model.clip_text_encoder, "text_encoder")
        ModelSaver.save_weights(base_path, bits, model.t5_text_encoder, "text_encoder_2")

    @staticmethod
    def _save_tokenizer(base_path: str, tokenizer: CLIPTokenizer | T5Tokenizer, subdir: str):
        path = Path(base_path) / subdir
        path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(path)

    @staticmethod
    def save_weights(base_path: str, bits: int, model: nn.Module, subdir: str):
        path = Path(base_path) / subdir
        path.mkdir(parents=True, exist_ok=True)
        weights = ModelSaver._split_weights(base_path, dict(tree_flatten(model.parameters())))
        for i, weight in enumerate(weights):
            mx.save_safetensors(
                str(path / f"{i}.safetensors"),
                weight,
                {"quantization_level": str(bits)},
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
