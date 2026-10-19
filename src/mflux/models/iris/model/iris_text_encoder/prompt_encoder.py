import mlx.core as mx
import numpy as np

from mflux.models.common.tokenizer import Tokenizer
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder


class IrisPromptEncoder:
    PREFIX = (
        "<|im_start|>system\n"
        "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial "
        "relationships of the objects and background:<|im_end|>\n"
        "<|im_start|>user\n"
    )
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
    MAX_LENGTH = 300
    HIDDEN_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)

    @staticmethod
    def tokenize(prompt: str, tokenizer: Tokenizer) -> tuple[mx.array, mx.array]:
        hf = tokenizer.tokenizer
        prefix_ids = hf.encode(IrisPromptEncoder.PREFIX, add_special_tokens=False)
        suffix_ids = hf.encode(IrisPromptEncoder.SUFFIX, add_special_tokens=False)
        budget = IrisPromptEncoder.MAX_LENGTH - len(suffix_ids)
        caption_ids = hf(prompt, add_special_tokens=False)["input_ids"][:budget]
        pad_id = hf.pad_token_id if hf.pad_token_id is not None else hf.eos_token_id

        total = len(prefix_ids) + IrisPromptEncoder.MAX_LENGTH
        real = prefix_ids + caption_ids + suffix_ids
        input_ids = np.full((1, total), pad_id, dtype=np.int32)
        attention_mask = np.zeros((1, total), dtype=np.int32)
        input_ids[0, : len(real)] = real
        attention_mask[0, : len(real)] = 1
        return mx.array(input_ids), mx.array(attention_mask)

    @staticmethod
    def encode_prompt(prompt: str, tokenizer: Tokenizer, text_encoder: Qwen3TextEncoder) -> tuple[mx.array, mx.array]:
        input_ids, attention_mask = IrisPromptEncoder.tokenize(prompt, tokenizer)
        start = input_ids.shape[1] - IrisPromptEncoder.MAX_LENGTH
        _, hidden_states = text_encoder(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        embeddings = mx.stack([hidden_states[i][:, start:] for i in IrisPromptEncoder.HIDDEN_LAYERS], axis=2)
        mask = attention_mask[:, start:]
        embeddings = embeddings * mask[:, :, None, None].astype(embeddings.dtype)
        return embeddings, mask
