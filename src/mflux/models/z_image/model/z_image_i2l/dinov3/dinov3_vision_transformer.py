import mlx.core as mx
import mlx.nn as nn

from mflux.models.z_image.model.z_image_i2l.dinov3.dinov3_embeddings import DINOv3Embeddings
from mflux.models.z_image.model.z_image_i2l.dinov3.dinov3_rope import compute_dinov3_rope_embeddings
from mflux.models.z_image.model.z_image_i2l.dinov3.dinov3_transformer_block import DINOv3TransformerBlock


class DINOv3VisionTransformer(nn.Module):
    """DINOv3-7B vision transformer for i2L image encoding.

    Config: hidden_size=4096, intermediate_size=8192, num_hidden_layers=40,
            num_attention_heads=32, image_size=224, patch_size=16,
            num_register_tokens=4, rope_theta=100.0.

    Input:  pixel_values (B, 3, 224, 224) normalized with ImageNet stats
    Output: pooled_output (B, 4096) — CLS token representation
    """

    def __init__(self):
        super().__init__()
        self.embeddings = DINOv3Embeddings()
        self.layer = [DINOv3TransformerBlock() for _ in range(40)]
        self.norm = nn.LayerNorm(dims=4096, eps=1e-5)

        # Constants for RoPE
        self.image_size = 224
        self.patch_size = 16
        self.num_prefix_tokens = 5  # 1 CLS + 4 registers

    def __call__(self, pixel_values: mx.array) -> mx.array:
        """
        Args:
            pixel_values: (B, 3, 224, 224)
        Returns:
            pooled_output: (B, 4096) — CLS token
        """
        # Compute RoPE embeddings for patch positions
        num_patches_h = self.image_size // self.patch_size  # 14
        num_patches_w = self.image_size // self.patch_size  # 14
        cos, sin = compute_dinov3_rope_embeddings(
            num_patches_h=num_patches_h,
            num_patches_w=num_patches_w,
            head_dim=128,
            theta=100.0,
            dtype=pixel_values.dtype,
        )

        # Embed patches + CLS + registers
        hidden_states = self.embeddings(pixel_values)

        # Transformer layers
        for layer_module in self.layer:
            hidden_states = layer_module(
                hidden_states,
                cos=cos,
                sin=sin,
                num_prefix_tokens=self.num_prefix_tokens,
            )

        # Final norm
        hidden_states = self.norm(hidden_states)

        # Return CLS token (index 0)
        pooled_output = hidden_states[:, 0, :]
        return pooled_output
