"""FLUX.2 VAE with 32 latent channels and 2x2 patches.

FLUX.2 VAE differs from FLUX.1:
- 32 latent channels (vs 16 in FLUX.1)
- 2x2 patch embedding (32 * 2 * 2 = 128 channels to transformer)
- Same encoder/decoder architecture (block_out_channels = [128, 256, 512, 512])
"""

import mlx.core as mx
from mlx import nn

from mflux.models.flux2.model.flux2_vae.decoder import Flux2Decoder
from mflux.models.flux2.model.flux2_vae.encoder import Flux2Encoder


class Flux2VAE(nn.Module):
    """FLUX.2 VAE with 32-channel latent space and patch embedding.

    The latent space is 32 channels at 1/8 spatial resolution.
    Patch embedding converts to 128 channels at 1/16 resolution for transformer.

    Attributes:
        scaling_factor: Scale factor for latent normalization
        shift_factor: Shift factor for latent normalization
        spatial_scale: Spatial downscale factor (8 for encoder output)
        latent_channels: Number of latent channels (32)
        patch_size: Size of patches for transformer input (2x2)
    """

    # Scaling factors verified for FLUX.2 32-channel VAE
    # These values are consistent with FLUX.1 despite the channel count difference
    # The scaling applies to the latent space normalization, not channel-specific
    # Reference: HuggingFace black-forest-labs/FLUX.2-dev AutoencoderKLFlux2
    scaling_factor: float = 0.3611
    shift_factor: float = 0.1159
    spatial_scale: int = 8
    latent_channels: int = 32
    patch_size: int = 2

    def __init__(self):
        super().__init__()
        self.decoder = Flux2Decoder()
        self.encoder = Flux2Encoder()

    def decode(self, latents: mx.array) -> mx.array:
        """Decode latents to image.

        Args:
            latents: Latent tensor [B, C, H, W] or [B, C, T, H, W]
                    After unpatchify: [B, 32, H//8, W//8]

        Returns:
            Decoded image [B, 3, T, H, W]
        """
        if latents.ndim == 5:
            latents = latents[:, :, 0, :, :]

        # Unpatchify if latents are in patched form (128 channels -> 32 channels)
        if latents.shape[1] == 128:
            latents = self._unpatchify(latents)

        scaled_latents = (latents / self.scaling_factor) + self.shift_factor
        decoded = self.decoder(scaled_latents)
        return decoded[:, :, None, :, :]

    def encode(self, image: mx.array) -> mx.array:
        """Encode image to latents.

        Args:
            image: Image tensor [B, 3, H, W] or [B, 3, T, H, W]

        Returns:
            Latent tensor [B, 32, T, H//8, W//8]
        """
        if image.ndim == 5:
            image = image[:, :, 0, :, :]

        latents = self.encoder(image)
        mean, _ = mx.split(latents, 2, axis=1)
        latent = (mean - self.shift_factor) * self.scaling_factor
        return latent[:, :, None, :, :]

    def patchify(self, latents: mx.array) -> mx.array:
        """Convert 32-channel latents to 128-channel patched form for transformer.

        Args:
            latents: Latent tensor [B, 32, H, W]

        Returns:
            Patched tensor [B, 128, H//2, W//2]
        """
        B, C, H, W = latents.shape
        assert C == self.latent_channels, f"Expected {self.latent_channels} channels, got {C}"

        # Reshape to extract 2x2 patches
        # [B, 32, H, W] -> [B, 32, H//2, 2, W//2, 2]
        latents = mx.reshape(latents, (B, C, H // 2, 2, W // 2, 2))

        # Permute to [B, 32, 2, 2, H//2, W//2]
        latents = mx.transpose(latents, (0, 1, 3, 5, 2, 4))

        # Flatten patches: [B, 32*2*2, H//2, W//2] = [B, 128, H//2, W//2]
        latents = mx.reshape(latents, (B, C * 4, H // 2, W // 2))

        return latents

    def _unpatchify(self, latents: mx.array) -> mx.array:
        """Convert 128-channel patched form back to 32-channel latents.

        Args:
            latents: Patched tensor [B, 128, H, W]

        Returns:
            Unpatched tensor [B, 32, H*2, W*2]
        """
        B, C, H, W = latents.shape
        assert C == self.latent_channels * 4, f"Expected {self.latent_channels * 4} channels, got {C}"

        # Reshape: [B, 128, H, W] -> [B, 32, 2, 2, H, W]
        latents = mx.reshape(latents, (B, self.latent_channels, 2, 2, H, W))

        # Permute to [B, 32, H, 2, W, 2]
        latents = mx.transpose(latents, (0, 1, 4, 2, 5, 3))

        # Reshape to [B, 32, H*2, W*2]
        latents = mx.reshape(latents, (B, self.latent_channels, H * 2, W * 2))

        return latents

    def latents_to_transformer_input(self, latents: mx.array) -> mx.array:
        """Prepare latents for transformer input.

        Takes encoded latents [B, 32, H//8, W//8] and converts to
        flattened sequence [B, (H//16)*(W//16), 128].

        Args:
            latents: Encoded latents [B, 32, H//8, W//8]

        Returns:
            Transformer input [B, seq_len, 128]
        """
        # Patchify: [B, 32, H//8, W//8] -> [B, 128, H//16, W//16]
        patched = self.patchify(latents)
        B, C, H, W = patched.shape

        # Flatten spatial dims and transpose: [B, H*W, C]
        flattened = mx.reshape(patched, (B, C, H * W))
        flattened = mx.transpose(flattened, (0, 2, 1))

        return flattened

    def transformer_output_to_latents(self, output: mx.array, height: int, width: int) -> mx.array:
        """Convert transformer output back to image latents.

        Args:
            output: Transformer output [B, seq_len, 128]
            height: Original image height
            width: Original image width

        Returns:
            Latents ready for decoder [B, 32, H//8, W//8]
        """
        B = output.shape[0]
        # Calculate spatial dimensions at 1/16 resolution
        H = height // 16
        W = width // 16

        # Transpose and reshape: [B, seq_len, 128] -> [B, 128, H, W]
        output = mx.transpose(output, (0, 2, 1))
        output = mx.reshape(output, (B, 128, H, W))

        # Unpatchify: [B, 128, H, W] -> [B, 32, H*2, W*2]
        latents = self._unpatchify(output)

        return latents
