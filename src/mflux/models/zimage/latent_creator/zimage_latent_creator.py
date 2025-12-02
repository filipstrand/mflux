import mlx.core as mx


class ZImageLatentCreator:
    @staticmethod
    def create_noise(seed: int, height: int, width: int) -> mx.array:
        """Create random noise for Z-Image latent space.

        Args:
            seed: Random seed
            height: Image height in pixels
            width: Image width in pixels

        Returns:
            Random noise tensor [1, 16, H/8, W/8]
        """
        return mx.random.normal(
            shape=[1, 16, height // 8, width // 8],
            key=mx.random.key(seed),
        )

    @staticmethod
    def pack_latents(latents: mx.array, height: int, width: int) -> mx.array:
        """Pack latents for Z-Image (identity operation).

        Z-Image uses unpacked latents in BCHW format [1, 16, H/8, W/8],
        unlike FLUX which uses packed format. So this is just an identity.

        Args:
            latents: Encoded latents from VAE [1, 16, H/8, W/8]
            height: Image height in pixels
            width: Image width in pixels

        Returns:
            Same latents unchanged
        """
        return latents
