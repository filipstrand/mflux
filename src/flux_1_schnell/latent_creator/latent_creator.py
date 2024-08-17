import mlx.core as mx


class LatentCreator:

    @staticmethod
    def create(height: int, width: int, seed: int) -> (mx.array, mx.array):
        latent_height = height // 16
        latent_width = width // 16
        latents = mx.random.normal(shape=[1, latent_height * latent_width, 64], key=mx.random.key(seed))
        return latents
