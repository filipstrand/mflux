import mlx.core as mx


class SeedVR2LatentCreator:
    @staticmethod
    def create_noise_latents(
        seed: int,
        height: int,
        width: int,
        batch_size: int = 1,
        latent_channels: int = 16,
    ) -> mx.array:
        return mx.random.normal(
            shape=(batch_size, latent_channels, 1, height, width),
            key=mx.random.key(seed),
        )

    @staticmethod
    def create_condition(encoded_latent: mx.array) -> mx.array:
        # Ensure we have 5D (B, C, T, H, W)
        if encoded_latent.ndim == 4:
            encoded_latent = encoded_latent[:, :, None, :, :]

        height = encoded_latent.shape[3]
        width = encoded_latent.shape[4]
        mask = mx.ones((1, 1, 1, height, width))
        condition_with_mask = mx.concatenate([encoded_latent, mask], axis=1)
        return condition_with_mask
