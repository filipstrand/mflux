import mlx.core as mx
import mlx.nn as nn


class PatchEmbed(nn.Module):
    """Patchify VAE latents into transformer tokens.

    Converts [B, C, H, W] latents into [B, N_patches, dim] sequence.
    Uses 2x2 patches on 16-channel VAE latents.
    """

    # Hardcoded architecture (mflux pattern)
    IN_CHANNELS = 16
    PATCH_SIZE = 2
    EMBED_DIM = 3840

    def __init__(self):
        super().__init__()
        # Input dimension: in_channels * patch_size^2 = 16 * 4 = 64
        in_dim = self.IN_CHANNELS * self.PATCH_SIZE * self.PATCH_SIZE
        self.proj = nn.Linear(in_dim, self.EMBED_DIM)

    def __call__(self, x: mx.array) -> mx.array:
        """Convert latents to patch tokens.

        Args:
            x: VAE latents [B, C, H, W] where C=16

        Returns:
            Patch tokens [B, N_patches, embed_dim] where N_patches = (H/2) * (W/2)
        """
        batch_size, channels, height, width = x.shape

        # Reshape into patches: [B, C, H/P, P, W/P, P]
        h_patches = height // self.PATCH_SIZE
        w_patches = width // self.PATCH_SIZE

        # Rearrange to [B, H/P, W/P, P*P*C] matching diffusers order (spatial first, then channel)
        # Diffusers: "c f pf h ph w pw -> (f h w) (pf ph pw c)"
        x = x.reshape(batch_size, channels, h_patches, self.PATCH_SIZE, w_patches, self.PATCH_SIZE)
        x = x.transpose(0, 2, 4, 3, 5, 1)  # [B, H/P, W/P, P_h, P_w, C]
        x = x.reshape(batch_size, h_patches * w_patches, -1)  # [B, N_patches, P*P*C]

        # Project to embedding dimension
        x = self.proj(x)

        return x
