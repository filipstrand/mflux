import mlx.core as mx

from mflux.models.flux_text_encoder.qwen_text_encoder import QwenTextEncoder
from mflux.tokenizer.qwen_tokenizer import TokenizerQwen


class QwenPromptEncoder:
    @staticmethod
    def encode_prompt(
        prompt: str,
        negative_prompt: str,
        prompt_cache: dict[str, tuple[mx.array, mx.array, mx.array, mx.array]],
        qwen_tokenizer: TokenizerQwen,
        qwen_text_encoder: QwenTextEncoder,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        # 0. Create a cache key that combines both prompts
        cache_key = f"{prompt}|NEG|{negative_prompt}"

        # 1. Return prompt encodings if already cached
        if cache_key in prompt_cache:
            return prompt_cache[cache_key]

        # 2. Encode the positive prompt
        input_ids, attention_mask = qwen_tokenizer.tokenize(prompt)
        neg_input_ids, neg_attention_mask = qwen_tokenizer.tokenize(negative_prompt)
        prompt_embeds, prompt_mask = qwen_text_encoder.encode(input_ids=input_ids, attention_mask=attention_mask, template_start_idx=0)
        neg_prompt_embeds, neg_prompt_mask = qwen_text_encoder.encode(input_ids=neg_input_ids, attention_mask=neg_attention_mask, template_start_idx=0)

        # 3. Cache the result (all 4 values)
        result = (prompt_embeds, prompt_mask, neg_prompt_embeds, neg_prompt_mask)
        prompt_cache[cache_key] = result
        return result
