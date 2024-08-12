import mlx.core as mx

from flux_1_schnell.config.config import Config


class FlowMatchEulerDiscreteNoiseScheduler:

    @staticmethod
    def denoise(
            t: int,
            noise: mx.array,
            latent: mx.array,
            config: Config,
    ) -> mx.array:
        sigma = config.sigmas[t]
        denoised = latent - noise * sigma
        derivative = (latent - denoised) / sigma
        dt = config.sigmas[t + 1] - sigma
        prev_sample = latent + derivative * dt
        return prev_sample
