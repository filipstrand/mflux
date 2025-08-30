from abc import ABC, abstractmethod

import mlx.core as mx


class BaseScheduler(ABC):
    """
    Abstract base class for all schedulers.
    """

    @property
    @abstractmethod
    def sigmas(self) -> mx.array:
        """
        The sigma schedule for the diffusion process.
        """
        raise NotImplementedError("Schedulers must implement sigmas")

    def scale_model_input(self, latents: mx.array, t: int) -> mx.array:
        """
        Scale the denoising model input. By default, no scaling applied.
        """
        return latents
