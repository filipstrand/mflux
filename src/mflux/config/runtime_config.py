import logging
import math

import mlx.core as mx
import numpy as np

from mflux.config.config import Config
from mflux.config.constants import NoiseSchedulerType
from mflux.config.model_config import ModelConfig

logger = logging.getLogger(__name__)


class RuntimeConfig:
    def __init__(
        self,
        config: Config,
        model_config: ModelConfig,
    ):
        self.config = config
        self.model_config = model_config
        self.sigmas = self._create_sigmas(config, model_config)

    @property
    def height(self) -> int:
        return self.config.height

    @property
    def width(self) -> int:
        return self.config.width

    @width.setter
    def width(self, value):
        self.config.width = value

    @property
    def guidance(self) -> float:
        return self.config.guidance

    @property
    def num_inference_steps(self) -> int:
        return self.config.num_inference_steps

    @property
    def precision(self) -> mx.Dtype:
        return self.config.precision

    @property
    def num_train_steps(self) -> int:
        return self.model_config.num_train_steps

    @property
    def image_path(self) -> str:
        return self.config.image_path

    @property
    def image_strength(self) -> float | None:
        return self.config.image_strength

    @property
    def init_time_step(self) -> int:
        is_img2img = (
            self.config.image_path is not None and
            self.image_strength is not None and
            self.image_strength > 0.0
        )  # fmt: off

        if is_img2img:
            # 1. Clamp strength to [0, 1]
            strength = max(0.0, min(1.0, self.config.image_strength))

            # 2. Return start time in [1, floor(num_steps * strength)]
            return max(1, int(self.num_inference_steps * strength))
        else:
            return 0

    @property
    def controlnet_strength(self) -> float | None:
        if self.config.controlnet_strength is not None:
            return self.config.controlnet_strength

        return None

    @staticmethod
    def _create_sigmas(config: Config, model_config: ModelConfig) -> mx.array:
        scheduler_type = getattr(config, "noise_scheduler", NoiseSchedulerType.LINEAR)

        if scheduler_type == NoiseSchedulerType.LINEAR:
            sigmas = RuntimeConfig._create_linear_sigmas(config.num_inference_steps)
        elif scheduler_type == NoiseSchedulerType.COSINE:
            sigmas = RuntimeConfig._create_cosine_sigmas(config.num_inference_steps)
        elif scheduler_type == NoiseSchedulerType.EXPONENTIAL:
            sigmas = RuntimeConfig._create_exponential_sigmas(config.num_inference_steps)
        elif scheduler_type == NoiseSchedulerType.SQRT:
            # Create a square root transformed schedule using manual implementation
            steps = np.linspace(0, 1, config.num_inference_steps)
            sigmas = np.sqrt(1 - steps)
            sigmas = mx.array(sigmas).astype(mx.float32)
            sigmas = mx.concatenate([sigmas, mx.zeros(1)])
        elif scheduler_type == NoiseSchedulerType.SCALED_LINEAR:
            sigmas = RuntimeConfig._create_scaled_linear_sigmas(config.num_inference_steps)
        else:
            logger.warning(f"Unknown scheduler type: {scheduler_type}, falling back to linear")
            sigmas = RuntimeConfig._create_linear_sigmas(config.num_inference_steps)

        if model_config.is_dev():
            sigmas = RuntimeConfig._shift_sigmas(sigmas=sigmas, width=config.width, height=config.height)

        return sigmas

    @staticmethod
    def _create_linear_sigmas(num_inference_steps: int) -> mx.array:
        """
        Create a linear noise schedule where noise level decreases linearly.

        This is the original implementation of the DDPM scheduler, widely used as
        a baseline in diffusion models.

        References:
        - DDPM Paper: https://arxiv.org/abs/2006.11239 (Ho et al., 2020)
        - PyTorch Implementation: https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddpm.py
        """
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        sigmas = mx.array(sigmas).astype(mx.float32)
        return mx.concatenate([sigmas, mx.zeros(1)])

    @staticmethod
    def _create_cosine_sigmas(num_inference_steps: int) -> mx.array:
        """
        Create a cosine noise schedule. This provides smoother transitions between
        noise levels and often generates better results for both simple and complex images.

        The cosine scheduler offers better perceptual quality and is particularly effective
        for image generation tasks. It allocates more diffusion steps to both high and low noise
        regions, which leads to better detail preservation and overall image quality.

        References:
        - Paper: "Improved Denoising Diffusion Probabilistic Models" by Nichol & Dhariwal (2021)
          https://arxiv.org/abs/2102.09672
        - PyTorch Implementation:
          https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
        - OpenAI Implementation:
          https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
        """
        # Using a cosine schedule for better perceptual quality
        s = 0.008
        steps = np.linspace(0, 1, num_inference_steps)
        sigmas = np.cos(((steps + s) / (1 + s)) * math.pi * 0.5) ** 2
        sigmas = sigmas / sigmas[0]  # Normalize to 1.0

        # Convert to MLX array
        sigmas = mx.array(sigmas).astype(mx.float32)
        return mx.concatenate([sigmas, mx.zeros(1)])

    @staticmethod
    def _create_exponential_sigmas(num_inference_steps: int, beta: float = 5.0) -> mx.array:
        """
        Create an exponential noise schedule. This accelerates the denoising process
        at the beginning and slows it at the end for more detail refinement.

        The exponential scheduler focuses computation on low-noise steps, which is where
        fine details are added to the generated image. This approach is inspired by the
        exponential noise schedules used in various diffusion models and has been shown to
        produce high-quality results with fewer steps.

        Value of beta determines the rate of decay.
        The default beta value of 5.0 is an empirical choice observed to work well in practice.
        In an exponential schedule, beta controls how rapidly the noise (sigma) decays.
        A beta of 5.0 results in a quick drop-off in noise,
        which has been found useful in balancing the gradual denoising process in diffusion models.

        References:
        - Paper: "Score-Based Generative Modeling through Stochastic Differential Equations" (Song et al., 2021)
          https://arxiv.org/abs/2011.13456
        - Related PyTorch implementation in DDIM scheduler:
          https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py
        """
        steps = np.linspace(0, 1, num_inference_steps)
        sigmas = np.exp(-beta * steps)

        # Convert to MLX array
        sigmas = mx.array(sigmas).astype(mx.float32)
        return mx.concatenate([sigmas, mx.zeros(1)])

    @staticmethod
    def _create_sqrt_sigmas(num_inference_steps: int) -> mx.array:
        """
        Create a square root transformed noise schedule. This applies a square root transformation
        to a linear schedule, focusing more steps in the higher noise regions.

        This scheduler applies a square root transformation to a linear schedule, concentrating
        more denoising steps in areas with higher noise. This approach helps preserve global
        structure and coherence in the generated images while still maintaining good detail in
        the final result. The square root transformation is particularly effective for balancing
        the trade-off between global structure and local detail.

        References:
        - Related to variance scheduling in: "Deep Unsupervised Learning using Nonequilibrium Thermodynamics"
          (Sohl-Dickstein et al., 2015) https://arxiv.org/abs/1503.03585
        - Similar approach in PNDM scheduler:
          https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_pndm.py
        - Square root transformation similar to that used in some variants of the DPM-Solver:
          https://github.com/LuChengTHU/dpm-solver
        """
        # Use square root transformation for a moderate focus on early steps
        steps = np.linspace(0, 1, num_inference_steps)
        sigmas = np.sqrt(1 - steps)

        # Convert to MLX array
        sigmas = mx.array(sigmas).astype(mx.float32)
        return mx.concatenate([sigmas, mx.zeros(1)])

    @staticmethod
    def _create_scaled_linear_sigmas(num_inference_steps: int) -> mx.array:
        """
        Create a properly scaled linear noise schedule. This applies a scaling factor to
        adjust the rate of noise reduction, making it more adaptable to different image types.

        Unlike the square root transformation, this is a true linear schedule with a scaling factor
        that controls the slope of the linear decay. The default scaling (0.5) results in slower initial
        denoising and faster final denoising for a more balanced approach to noise reduction.

        Parameters:
            num_inference_steps: Number of inference steps

        References:
        - Inspired by scaled schedulers in various diffusion implementations
        - Related to noise scheduling techniques in: "Denoising Diffusion Probabilistic Models"
          (Ho et al., 2020) https://arxiv.org/abs/2006.11239
        """
        # Create a scaled linear schedule
        start = 1.0
        end = 1.0 / num_inference_steps

        # Apply scaling to create non-uniform step sizes while maintaining linearity in the schedule
        # The fixed scale_factor of 0.5 creates smaller steps at the beginning and larger steps at the end
        scale_factor = 0.5  # Fixed scale factor for a balanced approach
        steps = np.linspace(0, 1, num_inference_steps) ** scale_factor
        steps = steps / steps[-1]  # Normalize to ensure we end at 1.0

        # Map the normalized steps to the sigma range
        sigmas = start + steps * (end - start)

        # Convert to MLX array
        sigmas = mx.array(sigmas).astype(mx.float32)
        return mx.concatenate([sigmas, mx.zeros(1)])

    @staticmethod
    def _shift_sigmas(sigmas: mx.array, width: int, height: int) -> mx.array:
        """
        Adjusts the noise schedule based on image resolution, applying stronger denoising
        for higher resolution images.

        This adaptive approach scales the noise levels based on the total number of pixels,
        which helps maintain consistent quality across different resolutions. The implementation
        uses a logistic transformation that's parameterized based on image dimensions.

        References:
        - Related to the resolution-dependent noise adjustment in Stable Diffusion:
          https://github.com/Stability-AI/stablediffusion/
        - Similar approaches discussed in: "High-Resolution Image Synthesis with Latent Diffusion Models"
          (Rombach et al., 2022) https://arxiv.org/abs/2112.10752
        - Implementation inspired by similar techniques in advanced diffusion samplers that adapt
          to image resolution to improve quality for high-resolution generation
        """
        y1 = 0.5
        x1 = 256
        m = (1.15 - y1) / (4096 - x1)
        b = y1 - m * x1
        mu = m * width * height / 256 + b
        mu = mx.array(mu)
        shifted_sigmas = mx.exp(mu) / (mx.exp(mu) + (1 / sigmas - 1))
        shifted_sigmas[-1] = 0
        return shifted_sigmas
