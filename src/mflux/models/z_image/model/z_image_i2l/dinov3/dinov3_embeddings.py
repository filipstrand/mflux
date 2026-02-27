import mlx.core as mx
import mlx.nn as nn


class DINOv3Embeddings(nn.Module):
    """DINOv3 embeddings: CLS token + register tokens + patch embeddings.

    image_size=224, patch_size=16, hidden_size=4096, num_register_tokens=4.
    Sequence: [CLS, reg0, reg1, reg2, reg3, patch0, patch1, ...]
    Total prefix tokens = 5 (1 CLS + 4 registers).
    """

    def __init__(self):
        super().__init__()
        self.hidden_size = 4096
        self.patch_size = 16
        self.image_size = 224
        self.num_register_tokens = 4

        self.cls_token = mx.random.normal(shape=(1, 1, self.hidden_size))
        self.register_tokens = mx.random.normal(shape=(1, self.num_register_tokens, self.hidden_size))
        self.patch_embeddings = nn.Conv2d(
            in_channels=3,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )

    def __call__(self, pixel_values: mx.array) -> mx.array:
        """
        Args:
            pixel_values: (B, 3, 224, 224)
        Returns:
            (B, 1 + 4 + 196, 4096) = (B, 201, 4096)
        """
        B = pixel_values.shape[0]

        # Patch embedding: (B, C, H, W) -> (B, H', W', hidden) via Conv2d (needs NHWC)
        x = mx.transpose(pixel_values, (0, 2, 3, 1))  # (B, H, W, C)
        x = self.patch_embeddings(x)  # (B, H', W', hidden)
        # Flatten spatial: (B, num_patches, hidden)
        x = x.reshape(B, -1, self.hidden_size)

        # Prepend CLS and register tokens
        cls_tokens = mx.broadcast_to(self.cls_token, (B, 1, self.hidden_size))
        reg_tokens = mx.broadcast_to(self.register_tokens, (B, self.num_register_tokens, self.hidden_size))
        x = mx.concatenate([cls_tokens, reg_tokens, x], axis=1)

        return x
