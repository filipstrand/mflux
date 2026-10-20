import mlx.core as mx


class IrisLatentCreator:
    PATCH_SIZE = 16
    CHANNELS = 3

    @staticmethod
    def create_noise(seed: int, height: int, width: int) -> mx.array:
        shape = (1, IrisLatentCreator.CHANNELS, height, width)
        return mx.random.normal(shape=shape, dtype=mx.float32, key=mx.random.key(seed))

    @staticmethod
    def unpack_latents(latents: mx.array, height: int, width: int) -> mx.array:
        return latents

    @staticmethod
    def check_size(height: int, width: int) -> None:
        if height % IrisLatentCreator.PATCH_SIZE or width % IrisLatentCreator.PATCH_SIZE:
            raise ValueError(f"width and height must be multiples of {IrisLatentCreator.PATCH_SIZE}")
