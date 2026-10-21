import mlx.core as mx
from mlx import nn

from mflux.models.moss_image.model.moss_image_transformer.moss_image_block import MOSSImageTextBlock


class MOSSImageTextProjection(nn.Module):
    def __init__(self, text_dim: int, dim: int, ffn_dim: int, heads: int, num_blocks: int = 2):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = [MOSSImageTextBlock(dim, ffn_dim, heads) for _ in range(num_blocks)]

    def __call__(self, x: mx.array, rope: mx.array) -> mx.array:
        x = self.proj(x)
        for block in self.blocks:
            x = block(x, rope)
        return x
