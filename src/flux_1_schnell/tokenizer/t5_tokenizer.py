import mlx.core as mx
from transformers import T5Tokenizer


class TokenizerT5:
    MAX_TOKEN_LENGTH = 256

    def __init__(self, tokenizer: T5Tokenizer):
        self.tokenizer = tokenizer

    def tokenize(self, prompt: str) -> mx.array:
        return self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=TokenizerT5.MAX_TOKEN_LENGTH,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="mlx",
        ).input_ids
