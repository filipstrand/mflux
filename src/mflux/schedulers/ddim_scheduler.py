from typing import Optional, Union

import mlx.core as mx


class DDIMScheduler:
    """
    DDIM scheduler for diffusion models.
    Provides deterministic sampling with fewer steps.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        eta: float = 0.0,
        set_alpha_to_one: bool = False,
        steps_offset: int = 0,
        prediction_type: str = "epsilon",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.eta = eta
        self.set_alpha_to_one = set_alpha_to_one
        self.steps_offset = steps_offset
        self.prediction_type = prediction_type

        # Create beta schedule
        if beta_schedule == "linear":
            self.betas = mx.linspace(beta_start, beta_end, num_train_timesteps, dtype=mx.float32)
        elif beta_schedule == "scaled_linear":
            # Common in DDPM/DDIM
            self.betas = mx.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=mx.float32) ** 2
        else:
            raise ValueError(f"Unsupported beta_schedule: {beta_schedule}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = mx.cumprod(self.alphas, axis=0)

        # Standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # Initialize timesteps
        self.timesteps = None
        self.num_inference_steps = None

    def set_timesteps(self, num_inference_steps: int, device: Optional[str] = None):
        """Set the discrete timesteps used for the diffusion chain."""
        self.num_inference_steps = num_inference_steps

        # Create timestep schedule
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = (mx.arange(0, num_inference_steps) * step_ratio).round().astype(mx.int32)
        timesteps = timesteps + self.steps_offset

        # Reverse for DDIM (from T to 0)
        self.timesteps = timesteps[::-1]

    def scale_model_input(self, sample: mx.array, timestep: Union[int, mx.array]) -> mx.array:
        """Scale the denoising model input."""
        return sample

    def step(
        self,
        model_output: mx.array,
        timestep: int,
        sample: mx.array,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator: Optional[int] = None,
        variance_noise: Optional[mx.array] = None,
        return_dict: bool = True,
    ) -> mx.array:
        """
        Predict the sample from the previous timestep by reversing the SDE.
        """
        # Get timestep indices
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps

        # Get alpha values
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else mx.array(1.0)

        beta_prod_t = 1 - alpha_prod_t

        # Compute predicted original sample
        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        elif self.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise ValueError(f"Unsupported prediction_type: {self.prediction_type}")

        # Clip predicted original sample
        if use_clipped_model_output:
            pred_original_sample = mx.clip(pred_original_sample, -1, 1)

        # Compute coefficients for pred_original_sample and current sample
        pred_original_sample_coeff = alpha_prod_t_prev**0.5
        current_sample_coeff = (1 - alpha_prod_t_prev) ** 0.5

        # Compute predicted previous sample
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * model_output

        # Add noise if eta > 0 (DDIM becomes stochastic)
        if eta > 0:
            # Compute variance
            variance = self._get_variance(timestep, prev_timestep)
            std_dev_t = eta * (variance**0.5)

            if variance_noise is None and generator is not None:
                mx.random.seed(generator)
                variance_noise = mx.random.normal(sample.shape, dtype=sample.dtype)
            elif variance_noise is None:
                variance_noise = mx.random.normal(sample.shape, dtype=sample.dtype)

            pred_prev_sample = pred_prev_sample + std_dev_t * variance_noise

        return pred_prev_sample

    def _get_variance(self, timestep: int, prev_timestep: int) -> mx.array:
        """Get the variance for the current timestep."""
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else mx.array(1.0)

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance
