import mlx.core as mx


class FiboLatentCreator:
    @staticmethod
    def create_noise(seed: int, height: int, width: int) -> mx.array:
        latents = mx.random.normal(
            shape=(1, 48, (height // 16), (width // 16)),
            key=mx.random.key(seed),
        )
        return FiboLatentCreator.pack_latents(latents, height, width)

    @staticmethod
    def pack_latents(latents: mx.array, height: int, width: int) -> mx.array:
        if latents.ndim == 5:
            latents = latents[:, :, 0, :, :]
        batch_size, channels, latent_height, latent_width = latents.shape
        latents = mx.transpose(latents, (0, 2, 3, 1))
        return mx.reshape(latents, (batch_size, latent_height * latent_width, channels))

    @staticmethod
    def unpack_latents(latents: mx.array, height: int, width: int) -> mx.array:
        batch_size, seq_len, channels = latents.shape
        vae_scale_factor = 16
        latent_height = height // vae_scale_factor
        latent_width = width // vae_scale_factor
        latents = mx.reshape(latents, (batch_size, latent_height, latent_width, channels))
        latents = mx.transpose(latents, (0, 3, 1, 2))
        return latents
