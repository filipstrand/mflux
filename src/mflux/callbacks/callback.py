from typing import Protocol

import mlx.core as mx
import PIL.Image
import tqdm

from mflux.config.runtime_config import RuntimeConfig


class BeforeLoopCallback(Protocol):
    def call_before_loop(
        self,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: RuntimeConfig,
        canny_image: PIL.Image.Image | None = None,
    ) -> None:  # fmt: off
        ...


class InLoopCallback(Protocol):
    def call_in_loop(
        self,
        t: int,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: RuntimeConfig,
        time_steps: tqdm
    ) -> None:  # fmt: off
        ...


class AfterLoopCallback(Protocol):
    def call_after_loop(
        self,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: RuntimeConfig
    ) -> None:  # fmt: off
        ...


class InterruptCallback(Protocol):
    def call_interrupt(
        self,
        t: int,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: RuntimeConfig,
        time_steps: tqdm
    ) -> None:  # fmt: off
        ...
