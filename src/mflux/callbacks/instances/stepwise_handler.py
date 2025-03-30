from pathlib import Path

import mlx.core as mx
import PIL.Image
import tqdm

from mflux.callbacks.callback import BeforeLoopCallback, InLoopCallback, InterruptCallback
from mflux.config.runtime_config import RuntimeConfig
from mflux.post_processing.array_util import ArrayUtil
from mflux.post_processing.image_util import ImageUtil


class StepwiseHandler(BeforeLoopCallback, InLoopCallback, InterruptCallback):
    def __init__(
        self,
        flux,
        output_dir: str,
    ):
        self.flux = flux
        self.output_dir = Path(output_dir)
        self.step_wise_images = []

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def call_before_loop(
        self,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: RuntimeConfig,
        canny_image: PIL.Image.Image | None = None,
        depth_image: PIL.Image.Image | None = None,
    ) -> None:
        self._save_image(
            step=config.init_time_step,
            seed=seed,
            prompt=prompt,
            latents=latents,
            config=config,
            time_steps=None,
        )

    def call_in_loop(
        self,
        t: int,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: RuntimeConfig,
        time_steps: tqdm,
    ) -> None:
        self._save_image(
            step=t + 1,
            seed=seed,
            prompt=prompt,
            latents=latents,
            config=config,
            time_steps=time_steps,
        )

    def call_interrupt(
        self,
        t: int,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: RuntimeConfig,
        time_steps: tqdm,
    ) -> None:
        self._save_composite(seed=seed)

    def _save_image(
        self,
        step: int,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: RuntimeConfig,
        time_steps: tqdm,
    ) -> None:
        unpack_latents = ArrayUtil.unpack_latents(latents=latents, height=config.height, width=config.width)
        stepwise_decoded = self.flux.vae.decode(unpack_latents)
        generation_time = time_steps.format_dict["elapsed"] if time_steps is not None else 0
        stepwise_img = ImageUtil.to_image(
            decoded_latents=stepwise_decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.flux.bits,
            lora_paths=self.flux.lora_paths,
            lora_scales=self.flux.lora_scales,
            generation_time=generation_time,
        )
        stepwise_img.save(
            path=self.output_dir / f"seed_{seed}_step{step}of{config.num_inference_steps}.png",
            export_json_metadata=False,
        )
        self.step_wise_images.append(stepwise_img)
        self._save_composite(seed=seed)

    def _save_composite(self, seed: int) -> None:
        if self.step_wise_images:
            composite_img = ImageUtil.to_composite_image(self.step_wise_images)
            composite_img.save(self.output_dir / f"seed_{seed}_composite.png")
