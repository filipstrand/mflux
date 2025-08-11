import logging
from pathlib import Path

import mlx.core as mx
import numpy as np

from mflux.config.config import Config
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

        # For Qwen models, timesteps are derived directly from sigmas (no complex scheduling needed)
        if "qwen" in model_config.model_name.lower():
            self.timesteps = True  # Just a flag to indicate Qwen model
        else:
            self.timesteps = None

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
    def image_path(self) -> Path | None:
        return self.config.image_path

    @property
    def image_strength(self) -> float | None:
        return self.config.image_strength

    @property
    def depth_image_path(self) -> Path | None:
        return self.config.depth_image_path

    @property
    def redux_image_paths(self) -> list[Path] | None:
        return self.config.redux_image_paths

    @property
    def redux_image_strengths(self) -> list[float] | None:
        return self.config.redux_image_strengths

    @property
    def masked_image_path(self) -> Path | None:
        return self.config.masked_image_path

    @property
    def init_time_step(self) -> int:
        is_img2img = (
            self.config.image_path is not None and
            self.image_strength is not None and
            self.image_strength > 0.0
        )  # fmt: off

        if is_img2img:
            # 1. Clamp strength to [0, 1]
            strength = max(0.0, min(1.0, self.config.image_strength))  # type: ignore

            # 2. Return start time in [1, floor(num_steps * strength)]
            return max(1, int(self.num_inference_steps * strength))  # type: ignore
        else:
            return 0

    @property
    def controlnet_strength(self) -> float | None:
        if self.config.controlnet_strength is not None:
            return self.config.controlnet_strength

        return None

    def get_qwen_timestep(self, t: int) -> float:
        """Get Qwen timestep value for given index.

        CRITICAL: This must return the exact same sigma values as the working version
        to maintain compatibility with the pretrained transformer weights.
        """
        if self.timesteps is not None:
            # Return the direct sigma value like the working version
            # Working version used: t_mx = mx.array(np.full((batch,), sigma, dtype=np.float32))
            # where sigma = sigmas[i] from linear sigmas
            return float(self.sigmas[t])
        else:
            raise ValueError("Timesteps not available for non-Qwen models")

    @staticmethod
    def _create_sigmas(config: Config, model_config: ModelConfig) -> mx.array:
        # Qwen uses different sigma calculation (Flow Matching)
        if "qwen" in model_config.model_name.lower():
            return RuntimeConfig._create_qwen_sigmas(config)

        # Standard Flux sigma calculation
        sigmas = RuntimeConfig._create_sigmas_values(config.num_inference_steps)
        if model_config.requires_sigma_shift:
            sigmas = RuntimeConfig._shift_sigmas(sigmas=sigmas, width=config.width, height=config.height)
        return sigmas

    @staticmethod
    def _create_sigmas_values(num_inference_steps: int) -> mx.array:
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        sigmas = mx.array(sigmas).astype(mx.float32)
        return mx.concatenate([sigmas, mx.zeros(1)])

    @staticmethod
    def _shift_sigmas(sigmas: mx.array, width: int, height: int) -> mx.array:
        y1 = 0.5
        x1 = 256
        m = (1.15 - y1) / (4096 - x1)
        b = y1 - m * x1
        mu = m * width * height / 256 + b
        mu = mx.array(mu)
        shifted_sigmas = mx.exp(mu) / (mx.exp(mu) + (1 / sigmas - 1))
        shifted_sigmas[-1] = 0
        return shifted_sigmas

    @staticmethod
    def _create_qwen_sigmas(config: Config) -> mx.array:
        """Create Flow Matching sigmas for Qwen models.

        CRITICAL: This must match the exact logic from the working version:
        sigmas = np.linspace(1.0, 1.0 / args.num_steps, args.num_steps, dtype=np.float32)
        """
        sigmas = np.linspace(1.0, 1.0 / config.num_inference_steps, config.num_inference_steps, dtype=np.float32)
        # Add zero at the end like Flux for dt calculation
        sigmas = np.append(sigmas, 0.0)
        return mx.array(sigmas).astype(mx.float32)
