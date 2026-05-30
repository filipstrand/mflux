from pathlib import Path

import mlx.core as mx
from mlx import nn
from PIL import Image

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.latent_creator.latent_creator import Img2Img, LatentCreator
from mflux.models.z_image.latent_creator import ZImageLatentCreator
from mflux.models.z_image.model.z_image_text_encoder.prompt_encoder import PromptEncoder
from mflux.models.z_image.variants.z_image import ZImage
from mflux.models.z_image.z_image_control_initializer import ZImageControlInitializer
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.image_util import ImageUtil


class ZImageControl(ZImage):
    """Z-Image-Turbo with the Fun-Controlnet-Union pose ControlNet (sc-2257).

    Strict pose conditioning: a rendered skeleton is VAE-encoded into the 33ch
    control context and threaded through the ported control branch
    (:class:`ZImageControlTransformer`). ``control_context_scale`` (recommended
    0.65–1.0) sets how hard the pose is locked; ``0.0`` reproduces base Z-Image.
    """

    def __init__(
        self,
        control_weights_path: str,
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        model_config: ModelConfig = ModelConfig.z_image_turbo(),
    ):
        nn.Module.__init__(self)
        ZImageControlInitializer.init(
            model=self,
            quantize=quantize,
            control_weights_path=control_weights_path,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            model_config=model_config,
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        control_image_path: Path | str,
        control_context_scale: float = 1.0,
        num_inference_steps: int = 8,
        height: int = 1024,
        width: int = 1024,
        guidance: float | None = None,
        scheduler: str | None = None,
        negative_prompt: str | None = None,
        image_path: Path | str | None = None,
        image_strength: float | None = None,
    ) -> Image.Image:
        supports_guidance = bool(self.model_config.supports_guidance)
        if not supports_guidance:
            guidance = 0.0
        if scheduler is None:
            scheduler = "flow_match_euler_discrete" if supports_guidance else "linear"

        config = Config(
            width=width,
            height=height,
            guidance=guidance,
            scheduler=scheduler,
            image_path=image_path,
            image_strength=image_strength,
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
        )

        # Initial latents
        latents = LatentCreator.create_for_txt2img_or_img2img(
            seed=seed,
            width=config.width,
            height=config.height,
            img2img=Img2Img(
                vae=self.vae,
                latent_creator=ZImageLatentCreator,
                image_path=config.image_path,
                sigmas=config.scheduler.sigmas,
                init_time_step=config.init_time_step,
                tiling_config=self.tiling_config,
            ),
        )

        # Control context: VAE-encode the rendered skeleton (already shift/scale
        # normalised by VAE.encode) into 16ch, then concat the zero mask (1ch) and
        # zero inpaint latent (16ch) → 33ch (C, F=1, H/8, W/8). Pure-pose control =
        # no init image / no mask, so those two channel groups are zeros.
        control_context = self._encode_control_context(control_image_path, config.width, config.height)

        text_encodings, negative_encodings = self._encode_prompts(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance=config.guidance,
        )

        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)
        predict = self._control_predict(control_context, control_context_scale)

        for t in config.time_steps:
            try:
                sigma_t = config.scheduler.sigmas[t].reshape((1,))
                timestep = mx.ones_like(sigma_t) - sigma_t
                noise = predict(
                    latents=latents,
                    timestep=timestep,
                    sigmas=config.scheduler.sigmas,
                    text_encodings=text_encodings,
                    negative_encodings=negative_encodings,
                    guidance=config.guidance,
                )
                latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)
                ctx.in_loop(t, latents)
                mx.eval(latents)
            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )

        ctx.after_loop(latents)
        decoded = self._decode_latents(latents=latents, config=config)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            image_path=config.image_path,
            image_strength=config.image_strength,
            generation_time=config.time_steps.format_dict["elapsed"],
            negative_prompt=negative_prompt,
        )

    def _encode_control_context(self, control_image_path: Path | str, width: int, height: int) -> mx.array:
        control_latents = LatentCreator.encode_image(
            vae=self.vae,
            image_path=control_image_path,
            height=height,
            width=width,
            tiling_config=self.tiling_config,
        )
        # Normalise to (16, 1, H/8, W/8)
        if control_latents.ndim == 4:  # (B, 16, h, w)
            control_latents = control_latents[0]
        elif control_latents.ndim == 5:  # (B, 16, 1, h, w)
            control_latents = control_latents[0, :, 0, :, :]
        control_latents = mx.expand_dims(control_latents, axis=1)  # (16, 1, h, w)

        c, _, h_lat, w_lat = control_latents.shape
        mask = mx.zeros((1, 1, h_lat, w_lat), dtype=control_latents.dtype)
        inpaint = mx.zeros((c, 1, h_lat, w_lat), dtype=control_latents.dtype)
        return mx.concatenate([control_latents, mask, inpaint], axis=0)  # (33, 1, h, w)

    def _control_predict(self, control_context: mx.array, control_context_scale: float):
        # No mx.compile: control_context is a constant captured per generation and
        # the Turbo path runs guidance 0 (single forward), so compilation buys
        # little and risks baking the control tensor.
        def predict(latents, timestep, sigmas, text_encodings, negative_encodings, guidance):
            noise = self.transformer(
                x=latents,
                timestep=timestep,
                sigmas=sigmas,
                cap_feats=text_encodings,
                control_context=control_context,
                control_context_scale=control_context_scale,
            )
            if negative_encodings is None:
                return noise
            negative_noise = self.transformer(
                x=latents,
                timestep=timestep,
                sigmas=sigmas,
                cap_feats=negative_encodings,
                control_context=control_context,
                control_context_scale=control_context_scale,
            )
            return noise + guidance * (noise - negative_noise)

        return predict
