import mlx.core as mx
from mlx import nn

from mflux.models.flux.model.flux_text_encoder.clip_encoder.clip_embeddings import CLIPEmbeddings
from mflux.models.flux.model.flux_text_encoder.clip_encoder.encoder_clip import EncoderCLIP


class CLIPTextModel(nn.Module):
    def __init__(self, dims: int, num_encoder_layers: int):
        super().__init__()
        self.encoder = EncoderCLIP(num_encoder_layers)
        self.embeddings = CLIPEmbeddings(dims)
        self.final_layer_norm = nn.LayerNorm(dims=768)

    def __call__(self, tokens: mx.array) -> tuple[mx.array, mx.array]:
        hidden_states = self.embeddings(tokens)
        causal_attention_mask = CLIPTextModel.create_causal_attention_mask(hidden_states.shape)
        encoder_outputs = self.encoder(hidden_states, causal_attention_mask)
        last_hidden_state = self.final_layer_norm(encoder_outputs)
        pooled_output = last_hidden_state[0, mx.argmax(tokens, axis=-1)]
        return pooled_output

    @staticmethod
    def create_causal_attention_mask(input_shape: tuple) -> mx.array:
        batch_size, query_length, _ = input_shape
        key_value_length = query_length
        mask = mx.tril(x=mx.ones((query_length, key_value_length)), k=0)
        mask = 1 - mask
        mask = mask * -3.4e38
        mask = mask.reshape((1, 1, query_length, key_value_length))
        mask = mx.broadcast_to(mask, (batch_size, 1, query_length, key_value_length))
        return mask
