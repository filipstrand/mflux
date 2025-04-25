import mlx.core as mx

from mflux.models.text_encoder.clip_encoder.clip_encoder import CLIPEncoder
from mflux.models.text_encoder.t5_encoder.t5_encoder import T5Encoder
from mflux.tokenizer.clip_tokenizer import TokenizerCLIP
from mflux.tokenizer.t5_tokenizer import TokenizerT5


class PromptEncoder:
    @staticmethod
    def encode_prompt(
        prompt: str,
        prompt_cache: dict[str, tuple[mx.array, mx.array]],
        t5_tokenizer: TokenizerT5,
        clip_tokenizer: TokenizerCLIP,
        t5_text_encoder: T5Encoder,
        clip_text_encoder: CLIPEncoder,
    ) -> tuple[mx.array, mx.array]:
        # 1. Return prompt encodings if already cached
        if prompt in prompt_cache:
            return prompt_cache[prompt]

        # 1. Encode the prompt
        t5_tokens = t5_tokenizer.tokenize(prompt)
        clip_tokens = clip_tokenizer.tokenize(prompt)
        prompt_embeds = t5_text_encoder(t5_tokens)
        pooled_prompt_embeds = clip_text_encoder(clip_tokens)

        # 2. Cache the encoded prompt
        prompt_cache[prompt] = (prompt_embeds, pooled_prompt_embeds)

        # 3. Return prompt encodings
        return prompt_embeds, pooled_prompt_embeds
