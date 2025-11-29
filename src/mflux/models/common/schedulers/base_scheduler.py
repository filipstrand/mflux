from abc import ABC, abstractmethod

import mlx.core as mx


class BaseScheduler(ABC):
    @property
    @abstractmethod
    def sigmas(self) -> mx.array: ...

    @abstractmethod
    def step(self, model_output: mx.array, timestep: int, sample: mx.array, **kwargs) -> mx.array: ...

    def scale_model_input(self, latents: mx.array, t: int) -> mx.array:
        return latents
