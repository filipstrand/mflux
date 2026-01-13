"""Latent creator for FLUX.2's 32-channel VAE.

FLUX.2 uses:
- 32 latent channels (vs 16 for FLUX.1)
- 2x2 patch embedding (same as FLUX.1)
- Resulting transformer input: 32 * 4 = 128 channels
"""

import mlx.core as mx


class Flux2LatentCreator:
    """Creates and manipulates latents for FLUX.2's 32-channel VAE."""

    @staticmethod
    def create_noise(seed: int, height: int, width: int) -> mx.array:
        """Create random noise latents for txt2img.

        Args:
            seed: Random seed for reproducibility
            height: Image height in pixels
            width: Image width in pixels

        Returns:
            Noise tensor [1, seq_len, 128] where seq_len = (h/16) * (w/16)
            128 = 32 latent channels * 4 (2x2 patches)
        """
        return mx.random.normal(
            shape=[1, (height // 16) * (width // 16), 128],
            key=mx.random.key(seed),
        )

    @staticmethod
    def pack_latents(latents: mx.array, height: int, width: int, num_channels_latents: int = 32) -> mx.array:
        """Pack latents from VAE format to transformer format.

        Converts [1, C, H, W] VAE output to [1, seq, C*4] transformer input.

        Args:
            latents: VAE latents [1, num_channels_latents, h/8, w/8]
            height: Original image height
            width: Original image width
            num_channels_latents: Number of VAE channels (32 for FLUX.2)

        Returns:
            Packed latents [1, (h/16)*(w/16), num_channels_latents*4]
        """
        # Reshape to expose 2x2 patches: [1, C, H/2, 2, W/2, 2]
        latents = mx.reshape(
            latents,
            (1, num_channels_latents, height // 16, 2, width // 16, 2)
        )
        # Permute to [1, H/2, W/2, C, 2, 2]
        latents = mx.transpose(latents, (0, 2, 4, 1, 3, 5))
        # Flatten to [1, seq, C*4]
        return mx.reshape(
            latents,
            (1, (width // 16) * (height // 16), num_channels_latents * 4)
        )

    @staticmethod
    def unpack_latents(latents: mx.array, height: int, width: int) -> mx.array:
        """Unpack latents from transformer format to VAE format.

        Converts [1, seq, 128] transformer output to [1, 32, h/8, w/8] VAE input.

        Args:
            latents: Packed latents [1, (h/16)*(w/16), 128]
            height: Original image height
            width: Original image width

        Returns:
            Unpacked latents [1, 32, h/8, w/8]
        """
        # Reshape to [1, H/2, W/2, 32, 2, 2]
        latents = mx.reshape(
            latents,
            (1, height // 16, width // 16, 32, 2, 2)
        )
        # Permute to [1, 32, H/2, 2, W/2, 2]
        latents = mx.transpose(latents, (0, 3, 1, 4, 2, 5))
        # Reshape to [1, 32, H, W] where H = h/8, W = w/8
        return mx.reshape(
            latents,
            (1, 32, height // 16 * 2, width // 16 * 2)
        )
