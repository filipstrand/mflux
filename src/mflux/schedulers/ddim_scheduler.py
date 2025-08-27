import math
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np


# Copied from diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar (lines 42-73)
def betas_for_alpha_bar(
    num_diffusion_timesteps,
    max_beta=0.999,
):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].
    """

    def alpha_bar_fn(t):
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return mx.array(betas, dtype=mx.float32)


# Ported from diffusers.schedulers.scheduling_ddim.rescale_zero_terminal_snr (lines 76-103)
def rescale_zero_terminal_snr(betas: mx.array) -> mx.array:
    """
    Rescales betas to have zero terminal SNR Based on https://huggingface.co/papers/2305.08891 (Algorithm 1)
    """
    # Convert betas to alphas_bar_sqrt
    alphas = 1.0 - betas
    alphas_cumprod = mx.cumprod(alphas, axis=0)
    alphas_bar_sqrt = mx.sqrt(alphas_cumprod)

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0]
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1]

    # Shift so the last timestep is zero.
    alphas_bar_sqrt_shifted = alphas_bar_sqrt - alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt_scaled = alphas_bar_sqrt_shifted * (alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T))

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt_scaled**2  # Revert sqrt
    alphas = alphas_bar[1:] / alphas_bar[:-1]  # Revert cumprod
    alphas = mx.concatenate([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas


class DDIMScheduler:
    """
    `DDIMScheduler` extends the denoising procedure introduced in denoising diffusion probabilistic models (DDPMs) with
    non-Markovian guidance.
    """

    def __init__(
        self,
        # Ported from diffusers.__init__ (lines 164-185)
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        clip_sample: bool = True,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
        prediction_type: str = "epsilon",
        clip_sample_range: float = 1.0,
        timestep_spacing: str = "leading",
        rescale_betas_zero_snr: bool = False,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.clip_sample = clip_sample
        self.set_alpha_to_one = set_alpha_to_one
        self.steps_offset = steps_offset
        self.prediction_type = prediction_type
        self.timestep_spacing = timestep_spacing
        self.rescale_betas_zero_snr = rescale_betas_zero_snr
        self.clip_sample_range = clip_sample_range

        # Ported from diffusers.__init__ (lines 187-196)
        if trained_betas is not None:
            self.betas = mx.array(trained_betas, dtype=mx.float32)
        elif beta_schedule == "linear":
            self.betas = mx.linspace(beta_start, beta_end, num_train_timesteps, dtype=mx.float32)
        elif beta_schedule == "scaled_linear":
            self.betas = mx.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=mx.float32) ** 2
        elif beta_schedule == "squaredcos_cap_v2":
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

        # Ported from diffusers.__init__ (lines 198-200)
        if rescale_betas_zero_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)

        # Ported from diffusers.__init__ (lines 201-203)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = mx.cumprod(self.alphas, axis=0)

        # Ported from diffusers.__init__ (lines 205-209)
        # At every step in ddim, we are looking into the previous alphas_cumprod
        # For the final step, there is no previous alphas_cumprod because we are already at 0
        # `set_alpha_to_one` decides whether we set this parameter simply to one or
        # whether we use the final alpha of the "non-previous" one.
        self.final_alpha_cumprod = mx.array(1.0) if set_alpha_to_one else self.alphas_cumprod[0]

        # Standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # Settable values
        self.num_inference_steps = None
        self.timesteps = mx.array(np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64))

    def scale_model_input(self, sample: mx.array, timestep: Optional[int] = None) -> mx.array:
        return sample

    # Ported from diffusers._get_variance (lines 268-275)
    def _get_variance(self, timestep, prev_timestep):
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance

    # Ported from diffusers.set_timesteps (lines 310-348)
    def set_timesteps(self, num_inference_steps: int):
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.num_train_timesteps`:"
                f" {self.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.num_train_timesteps} timesteps."
            )

        self.num_inference_steps = num_inference_steps

        if self.timestep_spacing == "linspace":
            timesteps = (
                np.linspace(0, self.num_train_timesteps - 1, num_inference_steps).round()[::-1].copy().astype(np.int64)
            )
        elif self.timestep_spacing == "leading":
            step_ratio = self.num_train_timesteps // self.num_inference_steps  # type: ignore[operator]
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
            timesteps += self.steps_offset
        elif self.timestep_spacing == "trailing":
            step_ratio = self.num_train_timesteps / self.num_inference_steps  # type: ignore[operator]
            timesteps = np.round(np.arange(self.num_train_timesteps, 0, -step_ratio)).astype(np.int64)
            timesteps -= 1
        else:
            raise ValueError(
                f"{self.timestep_spacing} is not supported. Please make sure to choose one of 'leading' or 'trailing'."
            )

        self.timesteps = mx.array(timesteps)

    # Ported from diffusers.step (lines 350-471), correcting the main algorithmic error
    def step(
        self,
        model_output: mx.array,
        timestep: int,
        sample: mx.array,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[mx.array] = None,
    ) -> Union[mx.array, Tuple]:
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # 1. get previous step value (=t-1)
        # See formulas (12) and (16) of DDIM paper https://arxiv.org/abs/2010.02502
        # (line 405)
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps

        # 2. compute alphas, betas
        # (lines 408-411)
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called "predicted x_0"
        # (lines 415-428)
        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        elif self.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.prediction_type} must be one of `epsilon`, `sample`, or `v_prediction`"
            )

        # 4. Clip "predicted x_0"
        # (lines 432-435)
        if self.clip_sample:
            pred_original_sample = mx.clip(pred_original_sample, -self.clip_sample_range, self.clip_sample_range)

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # (lines 438-439)
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance ** (0.5)

        if use_clipped_model_output:
            # the pred_epsilon is always re-derived from the clipped x_0 in Glide
            # (line 443)
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        # 6. compute "direction pointing to x_t" of formula (12)
        # (line 446)
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon

        # 7. compute x_t without "random noise" of formula (12)
        # (line 449)
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        if eta > 0:
            # (lines 451-464)
            if variance_noise is None:
                # MLX equivalent of randn_tensor
                variance_noise = mx.random.normal(model_output.shape, dtype=model_output.dtype)
            variance = std_dev_t * variance_noise

            prev_sample = prev_sample + variance

        return prev_sample, pred_original_sample

    # Copied from diffusers.add_noise (lines 473-496)
    def add_noise(
        self,
        original_samples: mx.array,
        noise: mx.array,
        timesteps: mx.array,
    ) -> mx.array:
        alphas_cumprod = self.alphas_cumprod.astype(original_samples.dtype)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = mx.expand_dims(sqrt_alpha_prod, -1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = mx.expand_dims(sqrt_one_minus_alpha_prod, -1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def __len__(self):
        return self.num_train_timesteps

    @property
    def sigmas(self) -> mx.array:
        # Calculate sigmas from alphas_cumprod (ascending order)
        sigmas_unflipped = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5
        # Reverse to get descending order, and append a zero for consistency
        return mx.concatenate([sigmas_unflipped[::-1], mx.array([0.0])])
