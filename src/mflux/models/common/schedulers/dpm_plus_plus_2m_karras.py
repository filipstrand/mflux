"""DPM++ 2M Karras scheduler for improved sampling quality.

DPM++ is a family of fast ODE solvers that achieve better quality than Euler
methods in fewer steps. The 2M variant uses second-order corrections for
improved accuracy.

The Karras noise schedule provides better step spacing for diffusion models,
with emphasis on the high-noise regime where detail decisions are made.

Benefits:
    - Better convergence in fewer steps (12-15 steps vs 20+ for Euler)
    - More consistent results across different prompts
    - Reduced color bleeding/saturation issues at high CFG

Reference:
    - "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models"
    - Karras et al. "Elucidating the Design Space of Diffusion-Based Generative Models"
"""

import math
from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from mflux.models.common.config.config import Config

from mflux.models.common.schedulers.base_scheduler import BaseScheduler


class DPMPlusPlusTwoMKarrasScheduler(BaseScheduler):
    """DPM++ 2M scheduler with Karras noise schedule.

    Implements the DPM-Solver++ algorithm with 2nd-order multistep correction
    and Karras-style sigma schedule for improved sampling quality.

    The Karras schedule places more steps in the high-noise regime (early steps)
    where semantic decisions are made, leading to better overall quality.

    Args:
        config: Configuration object with num_inference_steps and model_config

    Attributes:
        sigmas: Noise schedule following Karras formula
        timesteps: Step indices for the diffusion process
    """

    def __init__(self, config: "Config"):
        self.config = config
        self._sigmas = self._get_sigmas_karras()
        self._timesteps = self._get_timesteps()
        # Store previous model output for 2nd order correction
        self._prev_output: mx.array | None = None
        self._prev_timestep: int | None = None
        # Pre-allocate epsilon array to avoid repeated allocation in step()
        self._sigma_eps_array = mx.array(self._SIGMA_EPS)

    @property
    def sigmas(self) -> mx.array:
        return self._sigmas

    @property
    def timesteps(self) -> mx.array:
        return self._timesteps

    def _get_sigmas_karras(self) -> mx.array:
        """Generate Karras noise schedule.

        The Karras schedule uses the formula:
        sigma_i = (sigma_max^(1/rho) + i/(n-1) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho

        This places more steps in the high-noise regime where important
        semantic decisions are made.
        """
        num_steps = self.config.num_inference_steps

        # Karras schedule parameters
        sigma_min = 0.002  # Minimum noise level
        sigma_max = 80.0  # Maximum noise level
        rho = 7.0  # Controls step distribution (higher = more high-noise steps)

        # Apply resolution-aware sigma shift if required by model
        model_config = self.config.model_config
        if model_config.requires_sigma_shift:
            # Dynamic shift based on resolution (from linear scheduler)
            y1 = 0.5
            x1 = 256
            m = (1.15 - y1) / (4096 - x1)
            b = y1 - m * x1
            mu = m * self.config.width * self.config.height / 256 + b
            # Clamp mu to prevent overflow (max reasonable resolution ~16K x 16K)
            mu = min(mu, 10.0)
            # Scale sigma_max by the shift factor (use math.exp for scalar efficiency)
            sigma_max = sigma_max * math.exp(mu)

        # Generate Karras sigma schedule
        ramp = mx.linspace(0, 1, num_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)

        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        sigmas = mx.concatenate([sigmas, mx.zeros(1)])

        return sigmas.astype(mx.float32)

    def _get_timesteps(self) -> mx.array:
        """Get timestep indices for the schedule."""
        return mx.arange(self.config.num_inference_steps, dtype=mx.float32)

    # Minimum sigma threshold for numerical stability
    _SIGMA_EPS = 1e-10

    def step(
        self,
        noise: mx.array,
        timestep: int,
        latents: mx.array,
        **kwargs,
    ) -> mx.array:
        """Perform one DPM++ 2M step.

        Uses 2nd-order multistep correction when previous output is available.
        Falls back to 1st-order (Euler-like) on the first step.

        Args:
            noise: Model prediction (velocity or noise depending on parameterization)
            timestep: Current timestep index (must be in [0, num_steps - 1])
            latents: Current latent state

        Returns:
            Updated latents after one diffusion step

        Raises:
            ValueError: If timestep is out of bounds
        """
        # Bounds checking
        max_timestep = len(self._sigmas) - 2  # -2 because we access timestep + 1
        if timestep < 0 or timestep > max_timestep:
            raise ValueError(f"timestep {timestep} out of bounds [0, {max_timestep}]")

        sigma = self._sigmas[timestep]
        sigma_next = self._sigmas[timestep + 1]

        # Convert model output to denoised prediction (x0)
        # For flow matching: latents - sigma * noise = x0 estimate
        denoised = latents - sigma * noise

        # DPM++ 2M algorithm
        use_2and_order = self._prev_output is not None and self._prev_timestep is not None

        if use_2and_order:
            # 2nd order correction using previous output
            sigma_prev = self._sigmas[self._prev_timestep]

            # Safety check: avoid division by zero or near-zero
            # Use pre-allocated epsilon array to avoid allocation per step
            sigma_safe = mx.maximum(sigma, self._sigma_eps_array)
            sigma_prev_safe = mx.maximum(sigma_prev, self._sigma_eps_array)

            # Calculate coefficients for 2M correction
            h = mx.log(sigma_next / sigma_safe)
            h_prev = mx.log(sigma_safe / sigma_prev_safe)

            # Avoid division by zero in r calculation
            h_prev_safe = mx.where(
                mx.abs(h_prev) < self._SIGMA_EPS,
                self._sigma_eps_array,
                h_prev,
            )
            r = h / h_prev_safe

            # 2nd order Adams-Bashforth style correction
            d_cur = (denoised - latents) / (-sigma_safe)
            d_prev = (self._prev_output - latents) / (-sigma_prev_safe)

            # Weighted combination of current and previous derivatives
            d_prime = (1 + 0.5 / r) * d_cur - (0.5 / r) * d_prev

            # Euler step with corrected derivative
            dt = sigma_next - sigma
            latents_next = latents + d_prime * dt
        else:
            # 1st order (Euler) step - first step only
            dt = sigma_next - sigma
            latents_next = latents + noise * dt

        # Store for next step's 2nd order correction
        self._prev_output = denoised
        self._prev_timestep = timestep

        return latents_next

    def reset(self) -> None:
        """Reset scheduler state for new generation.

        Call this before starting a new image generation to clear
        the stored previous output.
        """
        self._prev_output = None
        self._prev_timestep = None

    def scale_model_input(self, latents: mx.array, t: int) -> mx.array:
        """Scale model input by sigma for CFG.

        DPM++ doesn't require input scaling, but we provide this
        for API compatibility.
        """
        return latents


class DPMPlusPlusSingleStepKarrasScheduler(BaseScheduler):
    """DPM++ 1M (single-step) scheduler with Karras schedule.

    Simpler variant without multistep correction. Useful for:
    - Debugging scheduler issues
    - Comparison benchmarks
    - When memory for prev_output is constrained

    Generally DPM++ 2M produces better results, but 1M is faster
    and has lower memory overhead.
    """

    def __init__(self, config: "Config"):
        self.config = config
        self._sigmas = self._get_sigmas_karras()
        self._timesteps = self._get_timesteps()

    @property
    def sigmas(self) -> mx.array:
        return self._sigmas

    @property
    def timesteps(self) -> mx.array:
        return self._timesteps

    def _get_sigmas_karras(self) -> mx.array:
        """Generate Karras noise schedule."""
        num_steps = self.config.num_inference_steps

        sigma_min = 0.002
        sigma_max = 80.0
        rho = 7.0

        model_config = self.config.model_config
        if model_config.requires_sigma_shift:
            y1 = 0.5
            x1 = 256
            m = (1.15 - y1) / (4096 - x1)
            b = y1 - m * x1
            mu = m * self.config.width * self.config.height / 256 + b
            # Clamp mu to prevent overflow (max reasonable resolution ~16K x 16K)
            mu = min(mu, 10.0)
            # Scale sigma_max by the shift factor (use math.exp for scalar efficiency)
            sigma_max = sigma_max * math.exp(mu)

        ramp = mx.linspace(0, 1, num_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)

        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        sigmas = mx.concatenate([sigmas, mx.zeros(1)])

        return sigmas.astype(mx.float32)

    def _get_timesteps(self) -> mx.array:
        return mx.arange(self.config.num_inference_steps, dtype=mx.float32)

    def reset(self) -> None:
        """Reset scheduler state for new generation.

        Single-step scheduler has no state to reset, but provided for API
        consistency with DPMPlusPlusTwoMKarrasScheduler.
        """
        pass

    def step(
        self,
        noise: mx.array,
        timestep: int,
        latents: mx.array,
        **kwargs,
    ) -> mx.array:
        """Perform one DPM++ single-step update.

        Args:
            noise: Model prediction
            timestep: Current timestep index (must be in [0, num_steps - 1])
            latents: Current latent state

        Returns:
            Updated latents after one diffusion step

        Raises:
            ValueError: If timestep is out of bounds
        """
        # Bounds checking
        max_timestep = len(self._sigmas) - 2
        if timestep < 0 or timestep > max_timestep:
            raise ValueError(f"timestep {timestep} out of bounds [0, {max_timestep}]")

        sigma = self._sigmas[timestep]
        sigma_next = self._sigmas[timestep + 1]

        # Simple Euler-like step with Karras schedule
        dt = sigma_next - sigma
        return latents + noise * dt
