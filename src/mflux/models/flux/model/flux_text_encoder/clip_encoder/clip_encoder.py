import mlx.core as mx
from mlx import nn

from mflux.models.flux.model.flux_text_encoder.clip_encoder.clip_text_model import CLIPTextModel


class CLIPEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_model = CLIPTextModel(dims=768, num_encoder_layers=12)

    def __call__(self, tokens: mx.array) -> mx.array:
        pooled_output = self.text_model(tokens)
        return pooled_output
