import mlx.core as mx
from transformers import T5Tokenizer


class TokenizerT5:
    MAX_TOKEN_LENGTH = 256

    def __init__(self, tokenizer: T5Tokenizer):
        self.tokenizer = tokenizer

    def tokenize(self, prompt: str) -> mx.array:
        text_input_ids = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=TokenizerT5.MAX_TOKEN_LENGTH,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        ).input_ids
        return mx.array(text_input_ids.cpu().numpy())
