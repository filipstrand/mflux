import mlx.core as mx

from mflux.models.z_image.model.z_image_text_encoder.text_encoder import TextEncoder
from mflux.models.z_image.tokenizer.tokenizer import Tokenizer


class PromptEncoder:
    @staticmethod
    def encode_prompt(
        prompt: str,
        tokenizer: Tokenizer,
        text_encoder: TextEncoder,
    ) -> mx.array:
        input_ids, attention_mask = tokenizer.encode(prompt)
        cap_feats = text_encoder(input_ids, attention_mask)
        num_valid = int(mx.sum(attention_mask[0]).item())
        return cap_feats[0, :num_valid, :]
