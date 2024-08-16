import mlx.core as mx


class LatentCreator:

    @staticmethod
    def create(height: int, width: int, seed: int) -> (mx.array, mx.array):
        r_h = height // 16
        r_w = width // 16
        latents = mx.random.normal(shape=[1, r_h * r_w, 64], key=mx.random.key(seed))
        return latents
