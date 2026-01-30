from pathlib import Path

import mlx.core as mx
from mlx import nn
from PIL import Image

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.latent_creator.latent_creator import Img2Img, LatentCreator
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.z_image.constants import MAX_BATCH_SIZE, MAX_PROMPT_LENGTH
from mflux.models.z_image.latent_creator import ZImageLatentCreator
from mflux.models.z_image.model.z_image_text_encoder.prompt_encoder import PromptEncoder
from mflux.models.z_image.model.z_image_text_encoder.text_encoder import TextEncoder
from mflux.models.z_image.model.z_image_transformer.transformer import ZImageTransformer
from mflux.models.z_image.model.z_image_vae.vae import VAE
from mflux.models.z_image.weights.z_image_weight_definition import ZImageWeightDefinition
from mflux.models.z_image.z_image_initializer import ZImageInitializer
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.image_util import ImageUtil


class ZImage(nn.Module):
    """Z-Image base model with full CFG (Classifier-Free Guidance) support.

    Unlike Z-Image-Turbo which uses guidance_scale=0, the base model supports:
    - CFG guidance (recommended: 3.0-5.0)
    - Negative prompts for better control
    - More inference steps (recommended: 28-50)
    - Higher diversity and fine-tunability
    """

    # Re-export constants for backwards compatibility and documentation
    MAX_PROMPT_LENGTH = MAX_PROMPT_LENGTH
    MAX_BATCH_SIZE = MAX_BATCH_SIZE

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
        compile_model: bool = True,
    ):
        super().__init__()
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
        # Compile transformer for faster inference (15-40% speedup)
        if compile_model:
            ZImageInitializer.compile_for_inference(self)

    def generate_image(
        self,
        seed: int,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        height: int = 1024,
        width: int = 1024,
        guidance_scale: float = 4.0,
        cfg_normalization: bool = False,
        image_path: Path | str | None = None,
        image_strength: float | None = None,
        scheduler: str = "linear",
    ) -> Image.Image:
        """Generate an image using Z-Image base model with CFG support.

        Args:
            seed: Random seed for reproducibility
            prompt: Text prompt describing the desired image
            negative_prompt: Text describing what to avoid in the image
            num_inference_steps: Number of denoising steps (recommended: 28-50)
            height: Output image height
            width: Output image width
            guidance_scale: CFG scale (recommended: 3.0-5.0)
            cfg_normalization: False for general stylism, True for realism
            image_path: Optional input image for img2img
            image_strength: Strength of input image influence (0-1)
            scheduler: Scheduler type
        """
        # 0. Create config with CFG enabled
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

        # 1. Create the initial latents
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

        # 2. Encode the prompt (positive)
        text_encodings = PromptEncoder.encode_prompt(
            prompt=prompt,
            tokenizer=self.tokenizers["z_image"],
            text_encoder=self.text_encoder,
        )

        # 3. Encode negative prompt for CFG
        if guidance_scale > 0 and negative_prompt is not None:
            negative_encodings = PromptEncoder.encode_prompt(
                prompt=negative_prompt if negative_prompt else "",
                tokenizer=self.tokenizers["z_image"],
                text_encoder=self.text_encoder,
            )
        else:
            negative_encodings = None

        # 4. Create callback context and call before_loop
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)

        for t in config.time_steps:
            try:
                # 5.t Predict the noise (with CFG if enabled)
                if guidance_scale > 0 and negative_encodings is not None:
                    # CFG: noise = uncond_noise + guidance_scale * (cond_noise - uncond_noise)
                    cond_noise = self.transformer(
                        t=t,
                        x=latents,
                        cap_feats=text_encodings,
                        sigmas=config.scheduler.sigmas,
                    )
                    uncond_noise = self.transformer(
                        t=t,
                        x=latents,
                        cap_feats=negative_encodings,
                        sigmas=config.scheduler.sigmas,
                    )

                    # Apply CFG normalization if requested (better for realism)
                    if cfg_normalization:
                        # Normalize the CFG output to preserve signal magnitude
                        cond_norm = mx.sqrt(mx.sum(cond_noise**2, axis=-1, keepdims=True) + 1e-8)
                        cfg_output = uncond_noise + guidance_scale * (cond_noise - uncond_noise)
                        cfg_norm = mx.sqrt(mx.sum(cfg_output**2, axis=-1, keepdims=True) + 1e-8)
                        # Prevent division by near-zero to avoid NaN outputs
                        cfg_norm = mx.maximum(cfg_norm, mx.array(1e-6))
                        noise = cfg_output * (cond_norm / cfg_norm)
                    else:
                        noise = uncond_noise + guidance_scale * (cond_noise - uncond_noise)
                else:
                    noise = self.transformer(
                        t=t,
                        x=latents,
                        cap_feats=text_encodings,
                        sigmas=config.scheduler.sigmas,
                    )

                # 6.t Take one denoise step
                latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)

                # 7.t Call subscribers in-loop
                ctx.in_loop(t, latents)

                # Evaluate to enable progress tracking (MLX lazy evaluation)
                mx.eval(latents)

            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )

        # 8. Call subscribers after loop
        ctx.after_loop(latents)

        # 9. Decode the latents and return the image
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

    def generate_images_sequential(
        self,
        seeds: list[int],
        prompts: list[str] | str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        height: int = 1024,
        width: int = 1024,
        guidance_scale: float = 4.0,
        cfg_normalization: bool = False,
        scheduler: str = "linear",
        verbose: bool = True,
    ) -> list[Image.Image]:
        """Generate multiple images sequentially with a convenient API.

        This is a convenience wrapper that generates images one at a time.
        It does NOT provide batched/parallel inference - use this when you
        need multiple images with different seeds/prompts but don't need
        true GPU parallelism.

        For true batched inference with 2-3x throughput improvement,
        batched transformer support would need to be implemented.

        Args:
            seeds: List of random seeds (one per image, max 64)
            prompts: Either a single prompt for all images, or a list of prompts
            negative_prompt: Text describing what to avoid (shared across batch)
            num_inference_steps: Number of denoising steps
            height: Output image height (same for all, 64-4096)
            width: Output image width (same for all, 64-4096)
            guidance_scale: CFG scale
            cfg_normalization: Whether to normalize CFG output
            scheduler: Scheduler type
            verbose: Whether to print progress messages (default: True)

        Returns:
            List of generated PIL Images

        Raises:
            ValueError: If inputs are invalid (empty seeds, mismatched counts,
                       dimensions out of range)

        Example:
            images = model.generate_images_sequential(
                seeds=[1, 2, 3, 4],
                prompts="a beautiful sunset",  # Same prompt for all
                num_inference_steps=50,
            )
        """
        # Validate seeds - fail fast before any processing
        if not isinstance(seeds, list) or len(seeds) == 0:
            raise ValueError("seeds must be a non-empty list")
        if len(seeds) > self.MAX_BATCH_SIZE:
            raise ValueError(f"Too many seeds ({len(seeds)}), maximum is {self.MAX_BATCH_SIZE}")
        # Validate each seed is an integer (fail fast before processing prompts)
        for i, seed in enumerate(seeds):
            if not isinstance(seed, int):
                raise TypeError(f"seed at index {i} must be an integer, got {type(seed).__name__}")

        batch_size = len(seeds)
        max_prompt_len = self.MAX_PROMPT_LENGTH  # Cache for loop

        # Handle prompt: either single or list
        if isinstance(prompts, str):
            if len(prompts) > max_prompt_len:
                raise ValueError(
                    f"Single prompt too long ({len(prompts)} chars), maximum is {max_prompt_len}. "
                    f"This prompt would be used for all {batch_size} images."
                )
            prompts = [prompts] * batch_size
        else:
            if len(prompts) != batch_size:
                raise ValueError(f"Number of prompts ({len(prompts)}) must match number of seeds ({batch_size})")
            # Validate each prompt length
            for i, p in enumerate(prompts):
                if not isinstance(p, str):
                    raise TypeError(f"Prompt at index {i} must be a string, got {type(p).__name__}")
                if len(p) > max_prompt_len:
                    raise ValueError(f"Prompt at index {i} too long ({len(p)} chars), maximum is {max_prompt_len}")

        # Generate each image sequentially
        # Note: This wrapper provides API convenience but not performance gains
        results = []
        for i, (seed, prompt) in enumerate(zip(seeds, prompts)):
            if verbose:
                print(f"Generating image {i + 1}/{batch_size} (seed={seed})")
            image = self.generate_image(
                seed=seed,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                cfg_normalization=cfg_normalization,
                scheduler=scheduler,
            )
            results.append(image)

        return results

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=ZImageWeightDefinition,
        )
