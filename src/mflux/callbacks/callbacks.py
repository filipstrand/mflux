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
        latents: mx.array,
        config: RuntimeConfig,
        canny_image: PIL.Image.Image | None = None,
        depth_image: PIL.Image.Image | None = None,
    ):
        for subscriber in CallbackRegistry.before_loop_callbacks():
            subscriber.call_before_loop(
                seed=seed,
                prompt=prompt,
                latents=latents,
                config=config,
                canny_image=canny_image,
                depth_image=depth_image,
            )

    @staticmethod
    def in_loop(
        t: int,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: RuntimeConfig,
        time_steps: tqdm,
    ):
        for subscriber in CallbackRegistry.in_loop_callbacks():
            subscriber.call_in_loop(
                t=t,
                seed=seed,
                prompt=prompt,
                latents=latents,
                config=config,
                time_steps=time_steps,
            )

    @staticmethod
    def after_loop(
        seed: int,
        prompt: str,
        latents: mx.array,
        config: RuntimeConfig,
    ):
        for subscriber in CallbackRegistry.after_loop_callbacks():
            subscriber.call_after_loop(seed=seed, prompt=prompt, latents=latents, config=config)

    @staticmethod
    def interruption(
        t: int,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: RuntimeConfig,
        time_steps: tqdm,
    ):
        for subscriber in CallbackRegistry.interrupt_callbacks():
            subscriber.call_interrupt(
                t=t,
                seed=seed,
                prompt=prompt,
                latents=latents,
                config=config,
                time_steps=time_steps,
            )
