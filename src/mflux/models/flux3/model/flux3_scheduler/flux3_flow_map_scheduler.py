import mlx.core as mx


class Flux3FlowMapScheduler:
    def __init__(self, num_inference_steps: int = 16, alpha: float = 6.93):
        self.num_inference_steps = num_inference_steps
        self.alpha = alpha
        t = mx.linspace(1.0, 0.0, num_inference_steps + 1, dtype=mx.float32)
        self.timesteps = alpha * t / (1.0 + (alpha - 1.0) * t)

    def step(self, latents: mx.array, velocity: mx.array, step: int) -> mx.array:
        # The flow map jumps from timesteps[step] to timesteps[step + 1] in one Euler update.
        dt = (self.timesteps[step + 1] - self.timesteps[step]).item()
        return latents + dt * velocity
