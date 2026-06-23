import mlx.core as mx

from mflux.models.common.tokenizer import Tokenizer
from mflux.models.krea2.model.krea2_text_encoder.text_encoder import Krea2TextEncoder


class Krea2PromptEncoder:
    PROMPT_PREFIX = (
        "<|im_start|>system\n"
        "Describe the image by detailing the color, shape, size, texture, quantity, text, "
        "spatial relationships of the objects and background:<|im_end|>\n"
        "<|im_start|>user\n"
    )
    PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
    PREFIX_START_IDX = 34
    SUFFIX_START_IDX = 5

    @staticmethod
    def encode_prompt(
        prompt: str,
        prompt_cache: dict[str, tuple[mx.array, mx.array]],
        tokenizer: Tokenizer,
        text_encoder: Krea2TextEncoder,
        max_length: int = 512,
    ) -> tuple[mx.array, mx.array]:
        cache_key = f"krea2:{max_length}:{prompt}"
        if cache_key in prompt_cache:
            return prompt_cache[cache_key]

        input_ids, attention_mask, position_ids = Krea2PromptEncoder._tokenize(
            prompt=prompt,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        hidden_states, mask = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            trim_start=Krea2PromptEncoder.PREFIX_START_IDX,
        )
        result = (hidden_states, mask)
        prompt_cache[cache_key] = result
        return result

    @staticmethod
    def _tokenize(prompt: str, tokenizer: Tokenizer, max_length: int) -> tuple[mx.array, mx.array, mx.array]:
        raw_tokenizer = tokenizer.tokenizer
        text = Krea2PromptEncoder.PROMPT_PREFIX + (prompt or "")
        suffix_inputs = raw_tokenizer(
            [Krea2PromptEncoder.PROMPT_SUFFIX],
            return_tensors="np",
        )
        inputs = raw_tokenizer(
            [text],
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            max_length=max_length + Krea2PromptEncoder.PREFIX_START_IDX - Krea2PromptEncoder.SUFFIX_START_IDX,
            return_tensors="np",
        )
        input_ids = mx.concatenate(
            [
                mx.array(inputs["input_ids"]),
                mx.array(suffix_inputs["input_ids"]),
            ],
            axis=1,
        )
        attention_mask = mx.concatenate(
            [
                mx.array(inputs["attention_mask"]),
                mx.array(suffix_inputs["attention_mask"]),
            ],
            axis=1,
        ).astype(mx.int32)
        position_ids = mx.cumsum(attention_mask, axis=1).astype(mx.int32) - 1
        position_ids = mx.maximum(position_ids, mx.zeros_like(position_ids))
        position_ids = mx.expand_dims(position_ids, axis=0)
        position_ids = mx.broadcast_to(position_ids, (3, position_ids.shape[1], position_ids.shape[2]))
        return input_ids, attention_mask, position_ids
