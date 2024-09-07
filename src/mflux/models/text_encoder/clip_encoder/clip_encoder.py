import mlx.core as mx
from mlx import nn

from mflux.models.text_encoder.clip_encoder.clip_text_model import CLIPTextModel


class CLIPEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.text_model = CLIPTextModel(dims=768, num_encoder_layers=12)

    def forward(self, tokens: mx.array) -> mx.array:
        pooled_output = self.text_model.forward(tokens)
        return pooled_output
