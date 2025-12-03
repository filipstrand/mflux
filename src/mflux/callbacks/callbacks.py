import mlx.core as mx
import PIL.Image
import tqdm

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config.config import Config


class Callbacks:
    @staticmethod
    def before_loop(
        registry: CallbackRegistry,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: Config,
        canny_image: PIL.Image.Image | None = None,
        depth_image: PIL.Image.Image | None = None,
    ):
        for subscriber in registry.before_loop_callbacks():
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
        registry: CallbackRegistry,
        t: int,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: Config,
        time_steps: tqdm,
    ):
        for subscriber in registry.in_loop_callbacks():
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
        registry: CallbackRegistry,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: Config,
    ):
        for subscriber in registry.after_loop_callbacks():
            subscriber.call_after_loop(seed=seed, prompt=prompt, latents=latents, config=config)

    @staticmethod
    def interruption(
        registry: CallbackRegistry,
        t: int,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: Config,
        time_steps: tqdm,
    ):
        for subscriber in registry.interrupt_callbacks():
            subscriber.call_interrupt(
                t=t,
                seed=seed,
                prompt=prompt,
                latents=latents,
                config=config,
                time_steps=time_steps,
            )
