import mlx.core as mx

from mflux.models.common.tokenizer import Tokenizer
from mflux.models.qwen.model.qwen_text_encoder.qwen_text_encoder import QwenTextEncoder


class QwenPromptEncoder:
    @staticmethod
    def encode_prompt(
        prompt: str,
        negative_prompt: str,
        prompt_cache: dict[str, tuple[mx.array, mx.array, mx.array, mx.array]],
        qwen_tokenizer: Tokenizer,
        qwen_text_encoder: QwenTextEncoder,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        # Use a space as fallback for empty negative prompt to ensure valid tokenization
        if not negative_prompt or not negative_prompt.strip():
            negative_prompt = " "

        # 0. Create a cache key that combines both prompts
        cache_key = f"{prompt}|NEG|{negative_prompt}"

        # 1. Return prompt encodings if already cached
        if cache_key in prompt_cache:
            return prompt_cache[cache_key]

        # 2. Encode the positive prompt
        pos_output = qwen_tokenizer.tokenize(prompt)
        prompt_embeds, prompt_mask = qwen_text_encoder(
            input_ids=pos_output.input_ids, attention_mask=pos_output.attention_mask
        )

        # 3. Encode the negative prompt
        neg_output = qwen_tokenizer.tokenize(negative_prompt)
        neg_prompt_embeds, neg_prompt_mask = qwen_text_encoder(
            input_ids=neg_output.input_ids, attention_mask=neg_output.attention_mask
        )

        # 4. Cache the result (all 4 values)
        result = (prompt_embeds, prompt_mask, neg_prompt_embeds, neg_prompt_mask)
        prompt_cache[cache_key] = result
        return result
