import mlx.core as mx
from mlx import nn

from mflux.models.qwen.model.qwen_text_encoder.qwen_encoder import QwenEncoder
from mflux.models.qwen.model.qwen_text_encoder_alternative.utils import process_text_embeddings_mlx


class QwenTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = QwenEncoder()

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        # Get hidden states from the encoder
        hidden_states = self.encoder(input_ids, attention_mask)

        # Apply the same post-processing as diffusers using the correct function
        prompt_embeds, encoder_attention_mask = process_text_embeddings_mlx(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            drop_idx=34,
            dtype=mx.bfloat16
        )

        return prompt_embeds, encoder_attention_mask
