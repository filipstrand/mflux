import mlx.core as mx
import numpy as np

from mflux.models.common.tokenizer.tokenizer import LanguageTokenizer
from mflux.models.krea2.model.krea2_text_encoder.krea2_text_encoder import Krea2TextEncoder

# Krea 2 (Qwen3-VL-4B) text-encoder taps and chat-template layout (from diffusers Krea2Pipeline).
TEXT_ENCODER_SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
PROMPT_TEMPLATE_PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
)
PROMPT_TEMPLATE_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
PROMPT_TEMPLATE_START_IDX = 34
PROMPT_TEMPLATE_NUM_SUFFIX_TOKENS = 5


class Krea2PromptEncoder:
    """Tokenizes prompts into the Krea 2 fixed-length layout and taps the selected encoder layers.

    Krea 2 pads in the *middle* of the chat template ([prefix | prompt | PAD | suffix]); the position
    ids count only valid tokens so the suffix gets the right mRoPE phase. The system prefix is dropped
    from the encoder outputs. Returns (hidden_states (B, seq, 12, 2560), attention_mask (B, seq) bool).
    """

    @staticmethod
    def encode_prompt(
        prompt: str | list[str],
        tokenizer: LanguageTokenizer,
        text_encoder: Krea2TextEncoder,
        max_sequence_length: int = 512,
    ) -> tuple[mx.array, mx.array]:
        hf_tok = tokenizer.tokenizer
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        prefix_idx = PROMPT_TEMPLATE_START_IDX

        text = [PROMPT_TEMPLATE_PREFIX + p for p in prompts]
        text_tokens = hf_tok(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_sequence_length + prefix_idx - PROMPT_TEMPLATE_NUM_SUFFIX_TOKENS,
            return_tensors="np",
        )
        suffix_tokens = hf_tok([PROMPT_TEMPLATE_SUFFIX] * len(text), return_tensors="np")

        input_ids = np.concatenate([text_tokens["input_ids"], suffix_tokens["input_ids"]], axis=1)
        attention_mask = np.concatenate(
            [text_tokens["attention_mask"], suffix_tokens["attention_mask"]], axis=1
        ).astype(bool)

        # Cumulative-valid-token positions (padding does not advance a position).
        position_ids = np.clip(np.cumsum(attention_mask.astype(np.int64), axis=-1) - 1, 0, None)

        hidden_states_list = text_encoder(
            input_ids=mx.array(input_ids.astype(np.int32)),
            attention_mask=mx.array(attention_mask.astype(np.int32)),
            position_ids=mx.array(position_ids.astype(np.int32)),
            output_hidden_states=True,
        )
        stacked = mx.stack([hidden_states_list[i] for i in TEXT_ENCODER_SELECT_LAYERS], axis=2)

        stacked = stacked[:, prefix_idx:]
        mask = mx.array(attention_mask[:, prefix_idx:].astype(np.int32))
        return stacked, mask
