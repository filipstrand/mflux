import math

import mlx.core as mx
import numpy as np

from mflux.models.common.tokenizer import Tokenizer
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder


class Flux3PromptEncoder:
    OUTPUT_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
    MAX_LENGTH = 4096
    PAD_MULTIPLE = 80

    @staticmethod
    def tokenize(prompt: str, tokenizer: Tokenizer) -> tuple[mx.array, mx.array]:
        # Qwen3-VL chat template, then right padding to the next multiple of 80 (reference behaviour).
        hf = tokenizer.tokenizer
        text = hf.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        cap = Flux3PromptEncoder.MAX_LENGTH
        real = len(hf(text, truncation=True, max_length=cap)["input_ids"])
        length = min(math.ceil(real / Flux3PromptEncoder.PAD_MULTIPLE) * Flux3PromptEncoder.PAD_MULTIPLE, cap)
        tokens = hf(text, return_tensors="np", padding="max_length", truncation=True, max_length=length)
        return mx.array(tokens["input_ids"]), mx.array(tokens["attention_mask"])

    @staticmethod
    def positions(attention_mask: mx.array) -> mx.array:
        # cumsum(mask) - 1, padded positions set to 1
        positions = mx.cumsum(attention_mask.astype(mx.int32), axis=-1) - 1
        return mx.where(attention_mask == 0, 1, positions)

    @staticmethod
    def encode_prompt(
        prompt: str,
        tokenizer: Tokenizer,
        text_encoder: Qwen3TextEncoder,
    ) -> tuple[mx.array, mx.array]:
        input_ids, attention_mask = Flux3PromptEncoder.tokenize(prompt, tokenizer)
        tokens = text_encoder.get_prompt_embeds(
            input_ids=input_ids,
            attention_mask=attention_mask,
            hidden_state_layers=Flux3PromptEncoder.OUTPUT_LAYERS,
            position_ids=Flux3PromptEncoder.positions(attention_mask),
        )
        return tokens.astype(mx.bfloat16), Flux3PromptEncoder.prepare_text_ids(tokens.shape[1])

    @staticmethod
    def prepare_text_ids(length: int) -> mx.array:
        # (0, 0, 0, position) with position = arange(L); float32 as the transformer expects.
        ids = np.zeros((length, 4), dtype=np.float32)
        ids[:, 3] = np.arange(length)
        return mx.array(ids)
