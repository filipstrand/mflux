import mlx.core as mx

from mflux.models.common.tokenizer import Tokenizer
from mflux.models.flux.model.flux_text_encoder.clip_encoder.clip_encoder import CLIPEncoder
from mflux.models.flux.model.flux_text_encoder.t5_encoder.t5_encoder import T5Encoder


class PromptEncoder:
    @staticmethod
    def encode_prompt(
        prompt: str,
        prompt_cache: dict[str, tuple[mx.array, mx.array]],
        t5_tokenizer: Tokenizer,
        clip_tokenizer: Tokenizer,
        t5_text_encoder: T5Encoder,
        clip_text_encoder: CLIPEncoder,
    ) -> tuple[mx.array, mx.array]:
        # 1. Return prompt encodings if already cached
        if prompt in prompt_cache:
            return prompt_cache[prompt]

        # 1. Encode the prompt
        t5_output = t5_tokenizer.tokenize(prompt)
        clip_output = clip_tokenizer.tokenize(prompt)
        prompt_embeds = t5_text_encoder(t5_output.input_ids)
        pooled_prompt_embeds = clip_text_encoder(clip_output.input_ids)

        # 2. Cache the encoded prompt
        prompt_cache[prompt] = (prompt_embeds, pooled_prompt_embeds)

        # 3. Return prompt encodings
        return prompt_embeds, pooled_prompt_embeds
