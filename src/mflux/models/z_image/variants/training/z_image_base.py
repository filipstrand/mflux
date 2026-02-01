"""Z-Image-Base model for training.

Z-Image-Base is the full non-distilled 6B model that supports:
- CFG (guidance_scale 3.0-5.0) for better control
- Negative prompts for artifact removal
- Full fine-tuning (no distillation to preserve)
- Higher diversity and stylistic range

Architecture:
- Transformer: S3-DiT (Scalable Single-Stream DiT), 30 layers
- Text Encoder: Qwen3-4B
- VAE: Flux-derived, 16-channel latent space
"""

from pathlib import Path

import mlx.core as mx
from mlx import nn
from PIL import Image

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.latent_creator.latent_creator import Img2Img, LatentCreator
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.z_image.latent_creator import ZImageLatentCreator
from mflux.models.z_image.model.z_image_text_encoder.prompt_encoder import PromptEncoder
from mflux.models.z_image.model.z_image_text_encoder.text_encoder import TextEncoder
from mflux.models.z_image.model.z_image_transformer.transformer import ZImageTransformer
from mflux.models.z_image.model.z_image_vae.vae import VAE
from mflux.models.z_image.z_image_initializer import ZImageInitializer
from mflux.utils.image_util import ImageUtil


class ZImageBase(nn.Module):
    """Z-Image-Base model for inference and training.

    Unlike Z-Image-Turbo, this model:
    - Uses CFG (classifier-free guidance)
    - Supports negative prompts
    - Requires more inference steps (typically 50)
    - Is designed for fine-tuning
    """

    vae: VAE
    text_encoder: TextEncoder
    transformer: ZImageTransformer

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        model_config: ModelConfig | None = None,
    ):
        super().__init__()
        # Use Z-Image base config (not turbo)
        if model_config is None:
            model_config = ModelConfig.z_image_base()

        ZImageInitializer.init(
            model=self,
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            model_config=model_config,
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
        height: int = 1024,
        width: int = 1024,
        image_path: Path | str | None = None,
        image_strength: float | None = None,
        scheduler: str = "linear",
    ) -> Image.Image:
        """Generate an image with CFG support.

        Args:
            seed: Random seed for reproducibility
            prompt: Text prompt describing the desired image
            negative_prompt: Text describing what to avoid (Z-Image-Base feature)
            num_inference_steps: Number of denoising steps (default 50 for Base)
            guidance_scale: CFG scale (3.0-5.0 recommended for Z-Image-Base)
            height: Image height
            width: Image width
            image_path: Optional init image for img2img
            image_strength: Strength of init image influence
            scheduler: Noise scheduler type
        """
        # Create config with CFG support
        config = Config(
            width=width,
            height=height,
            guidance=guidance_scale,
            scheduler=scheduler,
            image_path=image_path,
            image_strength=image_strength,
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
        )

        # Create initial latents
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

        # Encode prompts
        text_encodings = PromptEncoder.encode_prompt(
            prompt=prompt,
            tokenizer=self.tokenizers["z_image"],
            text_encoder=self.text_encoder,
        )

        # Encode negative prompt if using CFG
        negative_encodings = None
        if guidance_scale > 1.0 and negative_prompt:
            negative_encodings = PromptEncoder.encode_prompt(
                prompt=negative_prompt,
                tokenizer=self.tokenizers["z_image"],
                text_encoder=self.text_encoder,
            )

        # Start callbacks
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)

        for t in config.time_steps:
            # Predict noise (conditional)
            noise = self.transformer(
                t=t,
                x=latents,
                cap_feats=text_encodings,
                sigmas=config.scheduler.sigmas,
            )

            # Apply CFG if guidance > 1 and negative prompt provided
            if guidance_scale > 1.0 and negative_encodings is not None:
                noise_uncond = self.transformer(
                    t=t,
                    x=latents,
                    cap_feats=negative_encodings,
                    sigmas=config.scheduler.sigmas,
                )
                noise = noise_uncond + guidance_scale * (noise - noise_uncond)

            # Denoise step
            latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)

            # Callbacks
            ctx.in_loop(t, latents)
            mx.synchronize()

        ctx.after_loop(latents)

        # Decode latents to image
        latents = ZImageLatentCreator.unpack_latents(latents, config.height, config.width)
        decoded = VAEUtil.decode(vae=self.vae, latent=latents, tiling_config=self.tiling_config)

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
