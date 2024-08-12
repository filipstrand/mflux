import mlx.core as mx


class LatentCreator:

    @staticmethod
    def create(seed: int) -> (mx.array, mx.array):
        latents = mx.random.normal(shape=[1, 4096, 64], key=mx.random.key(seed))
        return latents
