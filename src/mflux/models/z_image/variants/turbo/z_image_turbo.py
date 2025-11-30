from pathlib import Path

import mlx.core as mx
from mlx import nn
from PIL import Image

from mflux.callbacks.callbacks import Callbacks
from mflux.config.model_config import ModelConfig
from mflux.config.runtime_config import RuntimeConfig
from mflux.models.common.latent_creator.latent_creator import Img2Img, LatentCreator
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.z_image.latent_creator import ZImageLatentCreator
from mflux.models.z_image.model.z_image_text_encoder.prompt_encoder import PromptEncoder
from mflux.models.z_image.model.z_image_text_encoder.text_encoder import TextEncoder
from mflux.models.z_image.model.z_image_transformer.transformer import ZImageTransformer
from mflux.models.z_image.model.z_image_vae.vae import VAE
from mflux.models.z_image.z_image_initializer import ZImageInitializer
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.image_util import ImageUtil


class ZImageTurbo(nn.Module):
    vae: VAE
    text_encoder: TextEncoder
    transformer: ZImageTransformer

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        quantize: int | None = None,
        local_path: str | None = None,
        load_text_encoder: bool = True,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ):
        super().__init__()
        ZImageInitializer.init(
            z_image_model=self,
            model_config=model_config or ModelConfig.z_image_turbo(),
            quantize=quantize,
            local_path=local_path,
            load_text_encoder=load_text_encoder,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int = 4,
        height: int = 1024,
        width: int = 1024,
        image_path: Path | str | None = None,
        image_strength: float | None = None,
        scheduler: str = "linear",
    ) -> Image.Image:
        # 0. Create a new runtime config based on the model type and input parameters
        config = RuntimeConfig(
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance=0.0,  # Turbo model uses no guidance
            image_path=image_path,
            image_strength=image_strength,
            scheduler=scheduler,
        )

        # 1. Create the initial latents
        latents = LatentCreator.create_for_txt2img_or_img2img(
            seed=seed,
            height=config.height,
            width=config.width,
            img2img=Img2Img(
                vae=self.vae,
                latent_creator=ZImageLatentCreator,
                image_path=config.image_path,
                sigmas=config.scheduler.sigmas,
                init_time_step=config.init_time_step,
            ),
        )

        # 2. Encode the prompt
        text_encodings = PromptEncoder.encode_prompt(
            prompt=prompt,
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
        )

        # (Optional) Call subscribers for beginning of loop
        Callbacks.before_loop(
            seed=seed,
            prompt=prompt,
            latents=latents,
            config=config,
        )

        for t in config.time_steps:
            try:
                # 3.t Predict the noise
                noise = self.transformer(
                    t=t,
                    x=latents,
                    cap_feats=text_encodings,
                    sigmas=config.scheduler.sigmas,
                )

                # 4.t Take one denoise step
                latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)

                # (Optional) Call subscribers in-loop
                Callbacks.in_loop(
                    t=t,
                    seed=seed,
                    prompt=prompt,
                    latents=latents,
                    config=config,
                    time_steps=config.time_steps,
                )

                # (Optional) Evaluate to enable progress tracking
                mx.eval(latents)

            except KeyboardInterrupt:  # noqa: PERF203
                Callbacks.interruption(
                    t=t,
                    seed=seed,
                    prompt=prompt,
                    latents=latents,
                    config=config,
                    time_steps=config.time_steps,
                )
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )

        # (Optional) Call subscribers after loop
        Callbacks.after_loop(
            seed=seed,
            prompt=prompt,
            latents=latents,
            config=config,
        )

        # 5. Decode the latents and return the image
        latents = ZImageLatentCreator.unpack_latents(latents, config.height, config.width)
        decoded = self.vae.decode(latents)
        mx.eval(decoded)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            generation_time=config.time_steps.format_dict["elapsed"],
        )

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            tokenizers=[
                ("tokenizer.tokenizer", "tokenizer"),
            ],
            components=[
                ("vae", "vae"),
                ("transformer", "transformer"),
                ("text_encoder", "text_encoder"),
            ],
        )
