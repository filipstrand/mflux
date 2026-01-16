import mlx.core as mx
import numpy as np


class FlowMatchEulerDiscreteScheduler:
    @staticmethod
    def get_timesteps_and_sigmas(
        image_seq_len: int,
        num_inference_steps: int,
        num_train_timesteps: int = 1000,
    ) -> tuple[mx.array, mx.array]:
        sigmas = mx.array(np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps), dtype=mx.float32)
        mu = FlowMatchEulerDiscreteScheduler._compute_empirical_mu(
            image_seq_len=image_seq_len,
            num_steps=num_inference_steps,
        )
        sigmas = FlowMatchEulerDiscreteScheduler._time_shift_exponential(mu, 1.0, sigmas)
        timesteps = sigmas * num_train_timesteps
        sigmas = mx.concatenate([sigmas, mx.zeros((1,), dtype=sigmas.dtype)], axis=0)
        return timesteps, sigmas

    @staticmethod
    def _compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
        a1, b1 = 8.73809524e-05, 1.89833333
        a2, b2 = 0.00016927, 0.45666666
        if image_seq_len > 4300:
            return float(a2 * image_seq_len + b2)
        m_200 = a2 * image_seq_len + b2
        m_10 = a1 * image_seq_len + b1
        a = (m_200 - m_10) / 190.0
        b = m_200 - 200.0 * a
        return float(a * num_steps + b)

    @staticmethod
    def _time_shift_exponential(mu: float, sigma_power: float, t: mx.array) -> mx.array:
        return mx.exp(mu) / (mx.exp(mu) + ((1.0 / t - 1.0) ** sigma_power))
