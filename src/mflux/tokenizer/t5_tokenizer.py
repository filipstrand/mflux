import mlx.core as mx
from transformers import T5Tokenizer


class TokenizerT5:

    def __init__(self, tokenizer: T5Tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def tokenize(self, prompt: str) -> mx.array:
        return self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="mlx",
        ).input_ids
