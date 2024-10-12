from pathlib import Path
from dataclasses import dataclass

import mlx.core as mx

from mflux.config.runtime_config import RuntimeConfig
from mflux.post_processing.array_util import ArrayUtil
from mflux.post_processing.generated_image import GeneratedImage
from mflux.post_processing.image_util import ImageUtil


@dataclass
class StepwiseOutput:
    image: GeneratedImage
    image_path: Path
    time_step: int

    def __str__(self):
        return f"Stepwise output: time_step={self.time_step} image_path={self.image_path}"


class StepwiseHandler:
    def __init__(
        self,
        flux,
        config: RuntimeConfig,
        seed: int,
        prompt: str,
        time_steps,
        output_dir: Path | None = None,
    ):
        self.flux = flux
        self.config = config
        self.seed = seed
        self.prompt = prompt
        self.output_dir = output_dir
        self.time_steps = time_steps
        self.step_wise_images = []

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_step(self, step: int, latents: mx.array):
        if self.output_dir:
            unpack_latents = ArrayUtil.unpack_latents(latents, self.config.height, self.config.width)
            stepwise_decoded = self.flux.vae.decode(unpack_latents)
            stepwise_img = ImageUtil.to_image(
                decoded_latents=stepwise_decoded,
                seed=self.seed,
                prompt=self.prompt,
                quantization=self.flux.bits,
                generation_time=self.time_steps.format_dict["elapsed"],
                lora_paths=self.flux.lora_paths,
                lora_scales=self.flux.lora_scales,
                config=self.config,
            )
            self.step_wise_images.append(stepwise_img)

            stepwise_img.save(
                path=(image_path := self.output_dir / f"seed_{self.seed}_step{step + 1}of{len(self.time_steps)}.png"),
                export_json_metadata=False,
            )
            return StepwiseOutput(image=stepwise_img, image_path=image_path, time_step=step)

    def handle_interruption(self):
        if self.step_wise_images:
            composite_img = ImageUtil.to_composite_image(self.step_wise_images)
            composite_img.save(self.output_dir / f"seed_{self.seed}_composite.png")
