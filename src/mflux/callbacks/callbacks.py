import mlx.core as mx
import PIL.Image
import tqdm

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.config.runtime_config import RuntimeConfig


class Callbacks:
    @staticmethod
    def before_loop(
        seed: int,
        prompt: str,
        canny_image: PIL.Image.Image | None = None,
    ):  # fmt: off
        for subscriber in CallbackRegistry.before_loop_callbacks():
            subscriber.call_before_loop(
                seed=seed,
                prompt=prompt,
                canny_image=canny_image,
            )

    @staticmethod
    def in_loop(
        seed: int,
        prompt: str,
        step: int,
        latents: mx.array,
        config: RuntimeConfig,
        time_steps: tqdm
    ):  # fmt: off
        for subscriber in CallbackRegistry.in_loop_callbacks():
            subscriber.call_in_loop(
                seed=seed,
                prompt=prompt,
                step=step,
                latents=latents,
                config=config,
                time_steps=time_steps,
            )

    @staticmethod
    def interruption(
        seed: int,
        prompt: str,
        step: int,
        latents: mx.array,
        config: RuntimeConfig,
        time_steps: tqdm
    ):  # fmt: off
        for subscriber in CallbackRegistry.interrupt_callbacks():
            subscriber.call_interrupt(
                seed=seed,
                prompt=prompt,
                step=step,
                latents=latents,
                config=config,
                time_steps=time_steps,
            )
