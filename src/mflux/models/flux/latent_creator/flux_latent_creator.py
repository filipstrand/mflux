import mlx.core as mx


class FluxLatentCreator:
    @staticmethod
    def create_noise(seed: int, height: int, width: int) -> mx.array:
        return mx.random.normal(
            shape=[1, (height // 16) * (width // 16), 64],
            key=mx.random.key(seed),
        )

    @staticmethod
    def pack_latents(latents: mx.array, height: int, width: int, num_channels_latents: int = 16) -> mx.array:
        latents = mx.reshape(latents, (1, num_channels_latents, height // 16, 2, width // 16, 2))
        latents = mx.transpose(latents, (0, 2, 4, 1, 3, 5))
        return mx.reshape(latents, (1, (width // 16) * (height // 16), num_channels_latents * 4))

    @staticmethod
    def unpack_latents(latents: mx.array, height: int, width: int) -> mx.array:
        latents = mx.reshape(latents, (1, height // 16, width // 16, 16, 2, 2))
        latents = mx.transpose(latents, (0, 3, 1, 4, 2, 5))
        return mx.reshape(latents, (1, 16, height // 16 * 2, width // 16 * 2))
