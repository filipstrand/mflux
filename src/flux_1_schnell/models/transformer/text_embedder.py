from mlx import nn
import mlx.core as mx


class TextEmbedder(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(768, 3072)
        self.linear_2 = nn.Linear(3072, 3072)

    def forward(self, caption: mx.array) -> mx.array:
        hidden_states = self.linear_1(caption)
        hidden_states = nn.silu(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states
