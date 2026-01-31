from pathlib import Path

import mlx.core as mx
from mlx import nn
from PIL import Image

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.latent_creator.latent_creator import Img2Img, LatentCreator
from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.z_image.constants import MAX_BATCH_SIZE, MAX_PROMPT_LENGTH
from mflux.models.z_image.latent_creator import ZImageLatentCreator
from mflux.models.z_image.model.z_image_text_encoder.prompt_encoder import PromptEncoder
from mflux.models.z_image.model.z_image_text_encoder.text_encoder import TextEncoder
from mflux.models.z_image.model.z_image_transformer.transformer import ZImageTransformer
from mflux.models.z_image.model.z_image_vae.vae import VAE
from mflux.models.z_image.optimization.prompt_cache import ZImagePromptCache
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
    prompt_cache: ZImagePromptCache

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
        enable_tiling: bool = False,
        enable_batched_cfg: bool = False,
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
            enable_tiling: Enable VAE tiling for high-resolution (4K+) generation
                          without running out of memory. Uses 8x8 tiles with overlap.
            enable_batched_cfg: Batch conditional/unconditional passes into single
                               transformer call for 30-50% CFG speedup. Requires
                               slightly more memory but reduces transformer calls.
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

        # 0.1 Enable tiling for high-resolution generation (4K+)
        # This prevents OOM by processing the VAE in tiles
        tiling_config = None
        if enable_tiling:
            tiling_config = TilingConfig(
                vae_decode_tiles_per_dim=8,
                vae_decode_overlap=8,
                vae_encode_tiled=True,
                vae_encode_tile_size=512,
                vae_encode_tile_overlap=64,
            )
        # Use instance tiling_config if set, otherwise use parameter
        effective_tiling_config = self.tiling_config if self.tiling_config is not None else tiling_config

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
                tiling_config=effective_tiling_config,
            ),
        )

        # 2. Encode the prompt (positive) - use cache for repeated prompts
        text_encodings = self.prompt_cache.get_or_compute(
            prompt=prompt,
            tokenizer=self.tokenizers["z_image"],
            text_encoder=self.text_encoder,
        )

        # 3. Encode negative prompt for CFG - also use cache
        if guidance_scale > 0 and negative_prompt is not None:
            negative_encodings = self.prompt_cache.get_or_compute(
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
                    if enable_batched_cfg:
                        # Batched CFG: Process cond and uncond in single transformer call
                        # This reduces overhead by ~30-50% at cost of 2x memory for latents
                        noise = self._compute_cfg_batched(
                            t=t,
                            latents=latents,
                            text_encodings=text_encodings,
                            negative_encodings=negative_encodings,
                            sigmas=config.scheduler.sigmas,
                            guidance_scale=guidance_scale,
                            cfg_normalization=cfg_normalization,
                        )
                    else:
                        # Sequential CFG: Two separate transformer calls
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
        decoded = VAEUtil.decode(vae=self.vae, latent=latents, tiling_config=effective_tiling_config)
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

    def _compute_cfg_batched(
        self,
        t: int,
        latents: mx.array,
        text_encodings: mx.array,
        negative_encodings: mx.array,
        sigmas: mx.array,
        guidance_scale: float,
        cfg_normalization: bool,
    ) -> mx.array:
        """Compute CFG noise with batched transformer call.

        Batches conditional and unconditional passes into a single transformer
        forward pass for 30-50% speedup. Uses slightly more memory (2x latents)
        but reduces transformer call overhead significantly.

        Args:
            t: Current timestep
            latents: Current latent representation
            text_encodings: Positive prompt embeddings
            negative_encodings: Negative prompt embeddings
            sigmas: Scheduler sigmas
            guidance_scale: CFG scale
            cfg_normalization: Whether to normalize CFG output

        Returns:
            CFG-guided noise prediction
        """
        # Batch latents: [cond_latents, uncond_latents]
        batched_latents = mx.concatenate([latents, latents], axis=0)

        # Batch embeddings: [positive, negative]
        batched_embeddings = mx.concatenate([text_encodings, negative_encodings], axis=0)

        # Single transformer call with batched inputs
        batched_noise = self.transformer(
            t=t,
            x=batched_latents,
            cap_feats=batched_embeddings,
            sigmas=sigmas,
        )

        # Split results back into conditional and unconditional
        cond_noise, uncond_noise = mx.split(batched_noise, 2, axis=0)

        # Apply CFG formula with optional normalization
        return ZImage._apply_cfg_formula(cond_noise, uncond_noise, guidance_scale, cfg_normalization)

    @staticmethod
    def _apply_cfg_formula(
        cond_noise: mx.array,
        uncond_noise: mx.array,
        guidance_scale: float,
        cfg_normalization: bool,
    ) -> mx.array:
        """Apply CFG formula with optional normalization.

        CFG formula: noise = uncond + guidance_scale * (cond - uncond)

        Args:
            cond_noise: Conditional noise prediction
            uncond_noise: Unconditional noise prediction
            guidance_scale: CFG scale factor
            cfg_normalization: Whether to normalize output to preserve signal magnitude

        Returns:
            Combined CFG noise prediction
        """
        cfg_output = uncond_noise + guidance_scale * (cond_noise - uncond_noise)

        if cfg_normalization:
            # Normalize the CFG output to preserve signal magnitude
            cond_norm = mx.sqrt(mx.sum(cond_noise**2, axis=-1, keepdims=True) + 1e-8)
            cfg_norm = mx.sqrt(mx.sum(cfg_output**2, axis=-1, keepdims=True) + 1e-8)
            # Prevent division by near-zero to avoid NaN outputs
            cfg_norm = mx.maximum(cfg_norm, mx.array(1e-6))
            return cfg_output * (cond_norm / cfg_norm)

        return cfg_output

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

    def generate_images_batched(
        self,
        seeds: list[int],
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        height: int = 1024,
        width: int = 1024,
        guidance_scale: float = 4.0,
        cfg_normalization: bool = False,
        scheduler: str = "linear",
        max_batch_size: int = 4,
    ) -> list[Image.Image]:
        """Generate multiple images with true batched inference.

        This method provides 2-3x throughput improvement by processing
        multiple images in parallel through the transformer. Unlike
        generate_images_sequential, this uses GPU parallelism for
        significant speedup.

        Constraints:
        - All images use the same prompt (batched embeddings)
        - All images have the same dimensions
        - Memory scales with batch size

        Args:
            seeds: List of random seeds (one per image)
            prompt: Single prompt for all images (batched processing)
            negative_prompt: Text describing what to avoid (shared across batch)
            num_inference_steps: Number of denoising steps
            height: Output image height (same for all)
            width: Output image width (same for all)
            guidance_scale: CFG scale
            cfg_normalization: Whether to normalize CFG output
            scheduler: Scheduler type
            max_batch_size: Maximum images to process in parallel (default: 4).
                           Larger batches use more memory but are faster.

        Returns:
            List of generated PIL Images

        Example:
            # Generate 8 variations of the same prompt
            images = model.generate_images_batched(
                seeds=[1, 2, 3, 4, 5, 6, 7, 8],
                prompt="a beautiful sunset over mountains",
                max_batch_size=4,  # Process 4 at a time
            )
        """
        if not isinstance(seeds, list) or len(seeds) == 0:
            raise ValueError("seeds must be a non-empty list")
        if len(seeds) > self.MAX_BATCH_SIZE:
            raise ValueError(f"Too many seeds ({len(seeds)}), maximum is {self.MAX_BATCH_SIZE}")
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if max_batch_size > self.MAX_BATCH_SIZE:
            raise ValueError(f"max_batch_size ({max_batch_size}) exceeds maximum ({self.MAX_BATCH_SIZE})")

        # Encode prompt once (shared across all batches)
        text_encodings = PromptEncoder.encode_prompt(
            prompt=prompt,
            tokenizer=self.tokenizers["z_image"],
            text_encoder=self.text_encoder,
        )

        # Encode negative prompt once if using CFG
        negative_encodings = None
        if guidance_scale > 0 and negative_prompt is not None:
            negative_encodings = PromptEncoder.encode_prompt(
                prompt=negative_prompt if negative_prompt else "",
                tokenizer=self.tokenizers["z_image"],
                text_encoder=self.text_encoder,
            )

        # Process seeds in batches
        all_images = []
        for batch_start in range(0, len(seeds), max_batch_size):
            batch_seeds = seeds[batch_start : batch_start + max_batch_size]
            batch_images = self._generate_batch(
                seeds=batch_seeds,
                text_encodings=text_encodings,
                negative_encodings=negative_encodings,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                cfg_normalization=cfg_normalization,
                scheduler=scheduler,
                prompt=prompt,
            )
            all_images.extend(batch_images)

        return all_images

    def _generate_batch(
        self,
        seeds: list[int],
        text_encodings: mx.array,
        negative_encodings: mx.array | None,
        num_inference_steps: int,
        height: int,
        width: int,
        guidance_scale: float,
        cfg_normalization: bool,
        scheduler: str,
        prompt: str,
    ) -> list[Image.Image]:
        """Generate a batch of images in parallel.

        Internal method for batched generation. All images in the batch
        share the same prompt embeddings and dimensions.
        """
        batch_size = len(seeds)

        # Create config
        config = Config(
            width=width,
            height=height,
            guidance=guidance_scale,
            scheduler=scheduler,
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
        )

        # Create batched latents (one per seed)
        batched_latents = []
        for seed in seeds:
            latent = LatentCreator.create_for_txt2img_or_img2img(
                seed=seed,
                width=config.width,
                height=config.height,
            )
            batched_latents.append(latent)

        # Stack into single batch tensor
        latents = mx.concatenate(batched_latents, axis=0)

        # Expand embeddings to batch size
        batched_text = mx.repeat(text_encodings, batch_size, axis=0)
        batched_negative = None
        if negative_encodings is not None:
            batched_negative = mx.repeat(negative_encodings, batch_size, axis=0)

        # Denoising loop
        for t in config.time_steps:
            if guidance_scale > 0 and batched_negative is not None:
                # CFG with batched processing
                noise = self._compute_cfg_batched_multi(
                    t=t,
                    latents=latents,
                    text_encodings=batched_text,
                    negative_encodings=batched_negative,
                    sigmas=config.scheduler.sigmas,
                    guidance_scale=guidance_scale,
                    cfg_normalization=cfg_normalization,
                )
            else:
                noise = self.transformer(
                    t=t,
                    x=latents,
                    cap_feats=batched_text,
                    sigmas=config.scheduler.sigmas,
                )

            latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)
            mx.synchronize()

        # Decode each latent separately and collect images
        images = []
        for i in range(batch_size):
            single_latent = latents[i : i + 1]
            unpacked = ZImageLatentCreator.unpack_latents(single_latent, config.height, config.width)
            decoded = VAEUtil.decode(vae=self.vae, latent=unpacked)
            image = ImageUtil.to_image(
                decoded_latents=decoded,
                config=config,
                seed=seeds[i],
                prompt=prompt,
                quantization=self.bits,
                lora_paths=self.lora_paths,
                lora_scales=self.lora_scales,
                generation_time=config.time_steps.format_dict["elapsed"],
            )
            images.append(image)

        return images

    def _compute_cfg_batched_multi(
        self,
        t: int,
        latents: mx.array,
        text_encodings: mx.array,
        negative_encodings: mx.array,
        sigmas: mx.array,
        guidance_scale: float,
        cfg_normalization: bool,
    ) -> mx.array:
        """Compute CFG for multiple images in a batch.

        Similar to _compute_cfg_batched but handles multiple images.
        Concatenates all conditional and unconditional passes.

        Args:
            t: Current timestep
            latents: Batched latents [B, C, H, W]
            text_encodings: Batched positive embeddings [B, T, D]
            negative_encodings: Batched negative embeddings [B, T, D]
            sigmas: Scheduler sigmas
            guidance_scale: CFG scale
            cfg_normalization: Whether to normalize

        Returns:
            CFG-guided noise for all images in batch [B, C, H, W]
        """
        batch_size = latents.shape[0]

        # Concatenate: [cond_batch, uncond_batch]
        batched_latents = mx.concatenate([latents, latents], axis=0)
        batched_embeddings = mx.concatenate([text_encodings, negative_encodings], axis=0)

        # Single transformer call for 2*batch_size samples
        batched_noise = self.transformer(
            t=t,
            x=batched_latents,
            cap_feats=batched_embeddings,
            sigmas=sigmas,
        )

        # Split back
        cond_noise = batched_noise[:batch_size]
        uncond_noise = batched_noise[batch_size:]

        # Apply CFG using shared helper
        return ZImage._apply_cfg_formula(cond_noise, uncond_noise, guidance_scale, cfg_normalization)

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=ZImageWeightDefinition,
        )
