import gc

import mlx.core as mx
import PIL.Image
from tqdm import tqdm

from mflux.callbacks.callback import AfterLoopCallback, BeforeLoopCallback, InLoopCallback
from mflux.config.runtime_config import RuntimeConfig


class MemorySaver(BeforeLoopCallback, InLoopCallback, AfterLoopCallback):
    """
    Optimizes memory usage by clearing caches and removing unused model
    components at strategic points in the execution cycle.
    """

    def __init__(self, flux, keep_transformer: bool = True, cache_limit_bytes: int = 1000**3):
        self.flux = flux
        self.keep_transformer = keep_transformer
        self.peak_memory: int = 0
        mx.set_cache_limit(cache_limit_bytes)
        mx.clear_cache()
        mx.reset_peak_memory()

    def call_before_loop(
        self,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: RuntimeConfig,
        canny_image: PIL.Image.Image | None = None,
        depth_image: PIL.Image.Image | None = None,
    ) -> None:
        self.peak_memory = mx.get_peak_memory()
        self._delete_encoders()

    def call_in_loop(
        self,
        t: int,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: RuntimeConfig,
        time_steps: tqdm,
    ) -> None:
        self.peak_memory = mx.get_peak_memory()

    def call_after_loop(
        self,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: RuntimeConfig,
    ) -> None:
        self.peak_memory = mx.get_peak_memory()
        if not self.keep_transformer:
            self._delete_transformer()

    def _delete_encoders(self) -> None:
        # repeated image generation only works with the same prompt (cache)
        self.flux.clip_text_encoder = None
        self.flux.t5_text_encoder = None
        gc.collect()
        mx.clear_cache()

    def _delete_transformer(self) -> None:
        self.flux.transformer = None
        if hasattr(self.flux, "transformer_controlnet"):
            self.flux.transformer_controlnet = None
        gc.collect()
        mx.clear_cache()

    def memory_stats(self) -> str:
        self.peak_memory = mx.get_peak_memory()
        return f"Peak MLX memory: {self.peak_memory / 10**9:.2f} GB"
