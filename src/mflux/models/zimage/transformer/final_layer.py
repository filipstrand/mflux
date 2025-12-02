import mlx.core as mx
import mlx.nn as nn


class FinalLayer(nn.Module):
    """Final output layer for S3-DiT.

    Projects transformer output back to latent space with AdaLN modulation.
    """

    DIM = 3840
    TEMB_DIM = 256  # Timestep embedding dimension (matches t_embedder output)
    PATCH_SIZE = 2
    IN_CHANNELS = 16  # VAE latent channels

    def __init__(self):
        super().__init__()

        # AdaLN modulation: takes 256-dim timestep embedding, outputs DIM for scale only
        # Maps to all_final_layer.2-1.adaLN_modulation.1.weight (SiLU applied before linear)
        # HF weight shape: (3840, 256) - only scale, no shift
        self.adaLN = nn.Linear(self.TEMB_DIM, self.DIM, bias=True)

        # Output projection (dim → patch_size² × in_channels)
        out_dim = self.PATCH_SIZE**2 * self.IN_CHANNELS  # 4 * 16 = 64
        self.linear = nn.Linear(self.DIM, out_dim, bias=True)

        # Layer norm WITHOUT learnable parameters (elementwise_affine=False in PyTorch)
        # Diffusers: nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm = nn.LayerNorm(self.DIM, eps=1e-6, affine=False)

    def __call__(self, x: mx.array, temb: mx.array) -> mx.array:
        """Project image tokens to patch outputs.

        Args:
            x: Image tokens [B, N_patches, dim]
            temb: Timestep embedding [B, temb_dim=256]

        Returns:
            Patch predictions [B, N_patches, patch_size² × in_channels]
        """
        # Get scale modulation (no shift in final layer)
        scale = self.adaLN(nn.silu(temb))  # [B, dim]
        scale = scale[:, None, :]  # [B, 1, dim]

        # Apply scale modulation and project
        x = self.norm(x) * (1 + scale)
        return self.linear(x)
