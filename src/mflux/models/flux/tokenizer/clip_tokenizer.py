import mlx.core as mx
from transformers import CLIPTokenizer


class TokenizerCLIP:
    MAX_TOKEN_LENGTH = 77

    def __init__(self, tokenizer: CLIPTokenizer):
        self.tokenizer = tokenizer

    def tokenize(self, prompt: str) -> mx.array:
        return self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=TokenizerCLIP.MAX_TOKEN_LENGTH,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="mlx",
        ).input_ids
