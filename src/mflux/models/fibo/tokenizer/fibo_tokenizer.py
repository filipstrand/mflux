import mlx.core as mx
import numpy as np
from transformers import PreTrainedTokenizer


class TokenizerFibo:
    def __init__(self, tokenizer: PreTrainedTokenizer, bot_token_id: int = 128000):
        self.tokenizer = tokenizer
        self.bot_token_id = bot_token_id

    def tokenize(
        self,
        prompts: list[str],
        max_length: int = 2048,
        padding: str = "longest",
        truncation: bool = True,
        add_special_tokens: bool = True,
    ) -> tuple[mx.array, mx.array]:
        prompts = [p if p is not None else "" for p in prompts]

        if all(p == "" for p in prompts):
            batch_size = len(prompts)
            input_ids_mx = mx.array(np.empty((batch_size, 0), dtype=np.int32))
            attention_mask_mx = mx.array(np.empty((batch_size, 0), dtype=np.int32))
        else:
            tokenized = self.tokenizer(
                prompts,
                padding=padding,
                max_length=max_length,
                truncation=truncation,
                add_special_tokens=add_special_tokens,
                return_tensors="mlx",
            )
            input_ids_mx = tokenized["input_ids"]
            attention_mask_mx = tokenized["attention_mask"]

        return input_ids_mx, attention_mask_mx
