import mlx.core as mx
from mlx import nn


class SiglipVisionEmbeddings(nn.Module):
    embed_dim = 1152
    image_size = 384
    patch_size = 14

    def __init__(self):
        super().__init__()
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def __call__(self, pixel_values: mx.array) -> mx.array:
        # Get embeddings
        pixel_values = mx.transpose(pixel_values, (0, 2, 3, 1))
        embeddings = self.patch_embedding(pixel_values)
        embeddings = mx.transpose(embeddings, (0, 3, 1, 2))
        embeddings = embeddings.reshape(1, 1152, 27 * 27)
        embeddings = mx.transpose(embeddings, (0, 2, 1))

        # Get position embeddings
        position_ids = mx.arange(self.num_positions)
        position_embeddings = self.position_embedding(position_ids)

        return embeddings + position_embeddings
