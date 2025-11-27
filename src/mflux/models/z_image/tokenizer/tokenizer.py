from pathlib import Path

import mlx.core as mx
from transformers import AutoTokenizer

from mflux.utils.download import snapshot_download


class Tokenizer:
    def __init__(self, tokenizer_path: str | Path):
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        self.max_length = 512

    @classmethod
    def from_pretrained(cls, repo_id: str = "Tongyi-MAI/Z-Image-Turbo", local_path: str | None = None) -> "Tokenizer":
        if local_path:
            tokenizer_path = Path(local_path) / "tokenizer"
        else:
            root_path = Path(snapshot_download(repo_id=repo_id, allow_patterns=["tokenizer/*"]))
            tokenizer_path = root_path / "tokenizer"
        return cls(tokenizer_path)

    def encode(self, prompt: str | list[str], max_length: int | None = None) -> tuple[mx.array, mx.array]:
        max_length = max_length or self.max_length
        if isinstance(prompt, str):
            prompt = [prompt]

        formatted_prompts = []
        for p in prompt:
            formatted = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            formatted_prompts.append(formatted)

        text_inputs = self.tokenizer(
            formatted_prompts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="np",
        )

        return mx.array(text_inputs.input_ids), mx.array(text_inputs.attention_mask)
