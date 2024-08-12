import mlx.core as mx
import mlx.nn as nn


class ConvNormOut(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.GroupNorm(
            num_groups=32,
            dims=512,
            eps=1e-6,
            affine=True,
            pytorch_compatible=True
        )

    def forward(self, input_array: mx.array) -> mx.array:
        input_array = mx.transpose(input_array, (0, 2, 3, 1))
        hidden_states = self.norm(input_array.astype(mx.float32)).astype(mx.float32)
        return mx.transpose(hidden_states, (0, 3, 1, 2))
