import mlx.core as mx
from mlx import nn


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 1152
        self.image_size = 384
        self.patch_size = 14

        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", mx.arange(self.num_positions).expand((1, -1)), persistent=False)

    def __call__(self, pixel_values: mx.array) -> mx.array:
        batch_size = pixel_values.shape[0]

        # Create patch embeddings (B, C, H, W) -> (B, hidden_size, H/patch_size, W/patch_size)
        embeddings = self.projection(pixel_values)

        # Flatten patches: (B, hidden_size, H/patch_size, W/patch_size) -> (B, hidden_size, num_patches)
        embeddings = embeddings.reshape(batch_size, -1, self.num_patches)

        # Transpose: (B, hidden_size, num_patches) -> (B, num_patches, hidden_size)
        embeddings = mx.transpose(embeddings, (0, 2, 1))

        # Add class token
        class_embeddings = mx.broadcast_to(self.class_embedding, (batch_size, 1, embeddings.shape[-1]))
        embeddings = mx.concatenate([class_embeddings, embeddings], axis=1)

        # Add position embeddings
        position_ids = mx.arange(embeddings.shape[1])
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = embeddings + position_embeddings

        return embeddings
