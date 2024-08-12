import mlx.core as mx

from flux_1_schnell.models.text_encoder.clip_encoder.clip_encoder import CLIPEncoder
from flux_1_schnell.models.text_encoder.t5_encoder.t5_encoder import T5Encoder
from flux_1_schnell.tokenizer.clip_tokenizer import TokenizerCLIP
from flux_1_schnell.tokenizer.t5_tokenizer import TokenizerT5


class TextEncoder:

    @staticmethod
    def encode(
            prompt: str,
            clip_tokenizer: TokenizerCLIP,
            t5_tokenizer: TokenizerT5,
            clip_text_encoder: CLIPEncoder,
            t5_text_encoder: T5Encoder
    ) -> (mx.array, mx.array):
        clip_tokens = clip_tokenizer.tokenize(prompt)
        pooled_prompt_embeds = clip_text_encoder.forward(clip_tokens)

        t5_tokens = t5_tokenizer.tokenize(prompt)
        prompt_embeds = t5_text_encoder.forward(t5_tokens)

        return prompt_embeds, pooled_prompt_embeds
