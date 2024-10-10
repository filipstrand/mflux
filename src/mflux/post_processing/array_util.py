import mlx.core as mx


class ArrayUtil:
    @staticmethod
    def unpack_latents(latents: mx.array, height: int, width: int) -> mx.array:
        latents = mx.reshape(latents, (1, height // 16, width // 16, 16, 2, 2))
        latents = mx.transpose(latents, (0, 3, 1, 4, 2, 5))
        latents = mx.reshape(latents, (1, 16, height // 16 * 2, width // 16 * 2))
        return latents

    @staticmethod
    def pack_latents(latents: mx.array, height: int, width: int) -> mx.array:
        latents = mx.reshape(latents, (1, 16, height // 16, 2, width // 16, 2))
        latents = mx.transpose(latents, (0, 2, 4, 1, 3, 5))
        latents = mx.reshape(latents, (1, (width // 16) * (height // 16), 64))
        return latents
