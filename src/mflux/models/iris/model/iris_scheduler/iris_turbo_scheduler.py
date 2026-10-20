import mlx.core as mx


class IrisTurboScheduler:
    SIGMAS = (1.0, 0.938, 0.833, 0.625)
    NUM_TRAIN_TIMESTEPS = 1000

    def __init__(self, sigmas: tuple[float, ...] = SIGMAS):
        self.sigmas = sigmas
        self.num_inference_steps = len(sigmas)

    def model_time(self, step: int) -> mx.array:
        return mx.array([self.sigmas[step]], dtype=mx.float32) * IrisTurboScheduler.NUM_TRAIN_TIMESTEPS

    def step(self, latents: mx.array, velocity: mx.array, step: int) -> tuple[mx.array, mx.array]:
        # Returns the next state and the clean prediction (the final image on the last step).
        sigma = mx.array(self.sigmas[step], dtype=mx.float32)
        clean = latents - sigma * velocity
        if step + 1 == self.num_inference_steps:
            return clean, clean
        implied_noise = (latents - (1 - sigma) * clean) / sigma
        next_sigma = mx.array(self.sigmas[step + 1], dtype=mx.float32)
        return (1 - next_sigma) * clean + next_sigma * implied_noise, clean
