import mlx.core as mx
from mlx import nn


class Siglip2VisionEmbeddings(nn.Module):
    """SigLIP2-G384 patch embeddings: image_size=384, patch_size=16, hidden_size=1536."""

    embed_dim = 1536
    image_size = 384
    patch_size = 16

    def __init__(self):
        super().__init__()
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2  # 576
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

    def __call__(self, pixel_values: mx.array) -> mx.array:
        # pixel_values: (B, C, H, W) -> need (B, H, W, C) for MLX Conv2d
        x = mx.transpose(pixel_values, (0, 2, 3, 1))
        x = self.patch_embedding(x)
        # Output: (B, H', W', embed_dim) -> flatten spatial to (B, N, embed_dim)
        x = mx.transpose(x, (0, 3, 1, 2))  # (B, embed_dim, H', W')
        B = x.shape[0]
        num_patches_h = self.image_size // self.patch_size  # 24
        x = x.reshape(B, self.embed_dim, num_patches_h * num_patches_h)
        x = mx.transpose(x, (0, 2, 1))  # (B, N, embed_dim)

        position_ids = mx.arange(self.num_patches)
        x = (
            x + self.position_embedding(position_ids)[None, :, :]
        )  # (1, N, embed_dim) -> broadcast with (B, N, embed_dim)
        return x
