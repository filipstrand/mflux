import mlx.core as mx
from mlx import nn

from mflux.models.text_encoder.t5_encoder.t5_block import T5Block
from mflux.models.text_encoder.t5_encoder.t5_layer_norm import T5LayerNorm


class T5Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.shared = nn.Embedding(num_embeddings=32128, dims=4096)
        self.t5_blocks = [T5Block(i) for i in range(24)]
        self.final_layer_norm = T5LayerNorm()

    def forward(self, tokens: mx.array):
        hidden_states = self.shared(tokens)
        for block in self.t5_blocks:
            hidden_states = block.forward(hidden_states)
        hidden_states = self.final_layer_norm.forward(hidden_states)
        return hidden_states
