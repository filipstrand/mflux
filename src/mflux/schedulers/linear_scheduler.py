from typing import Optional, Union

import mlx.core as mx


class LinearScheduler:
    """
    Linear scheduler - the default scheduler used in mflux.
    Creates a linear schedule from 1.0 to 1/num_steps.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting: bool = False,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.use_dynamic_shifting = use_dynamic_shifting
        self.timesteps = None
        self.sigmas = None

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Optional[str] = None,
        sigmas: Optional[mx.array] = None,
        mu: Optional[float] = None,
    ):
        """Set the discrete timesteps used for the diffusion chain."""
        self.num_inference_steps = num_inference_steps

        if sigmas is not None:
            self.sigmas = sigmas
        else:
            # Default FLUX sigma schedule
            self.sigmas = self._get_sigmas(num_inference_steps, mu)

        self.timesteps = mx.arange(num_inference_steps, dtype=mx.int32)

    def _get_sigmas(self, num_inference_steps: int, mu: Optional[float] = None) -> mx.array:
        """Generate sigma schedule matching original mflux implementation."""
        # Create linear schedule from 1.0 to 1/num_steps
        sigmas = mx.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)

        # Add final sigma of 0
        sigmas = mx.concatenate([sigmas, mx.zeros(1)])

        return sigmas

    def scale_model_input(
        self, sample: mx.array, timestep: Union[float, mx.array], noise: Optional[mx.array] = None
    ) -> mx.array:
        """Scale the denoising model input."""
        return sample

    def step(
        self,
        model_output: mx.array,
        timestep: int,
        sample: mx.array,
        generator: Optional[int] = None,
        return_dict: bool = True,
    ) -> mx.array:
        """
        Predict the sample from the previous timestep by reversing the SDE.
        """
        if timestep >= len(self.sigmas) - 1:
            return sample

        # Get current and next sigma values
        sigma_current = self.sigmas[timestep]
        sigma_next = self.sigmas[timestep + 1]

        # Euler step
        dt = sigma_next - sigma_current
        sample = sample + model_output * dt

        return sample
