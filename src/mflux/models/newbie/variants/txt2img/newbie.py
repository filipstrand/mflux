"""
NewBie-image text-to-image generation model.

NewBie-image is a 3.5B parameter model optimized for anime/illustration generation.
Features:
- NextDiT transformer with Grouped Query Attention (GQA)
- Dual text encoders: Gemma3-4B-it + Jina CLIP v2
- 16-channel VAE (FLUX.1-dev)
- AdaLN-Single conditioning
"""

import mlx.core as mx
from mlx import nn
from PIL import Image

from mflux.models.common.config import ModelConfig
from mflux.models.common.schedulers.flow_match_euler_discrete_scheduler import FlowMatchEulerDiscreteScheduler
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.newbie.newbie_initializer import NewBieInitializer
from mflux.models.newbie.weights.newbie_weight_definition import NewBieWeightDefinition


class NewBie(nn.Module):
    """
    NewBie-image text-to-image generation model.

    A 3.5B parameter diffusion model using:
    - NextDiT transformer with GQA (36 blocks)
    - Dual text encoders (Gemma3 + Jina CLIP)
    - 16-channel VAE
    - Flow Match sampling

    Args:
        model_config: Model configuration
        quantize: Quantization bit width (4 or 8) or None for full precision
        model_path: Path to model weights
        lora_paths: List of LoRA paths to load
        lora_scales: List of LoRA scales (default 1.0)
    """

    def __init__(
        self,
        model_config: ModelConfig = ModelConfig.newbie(),
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ):
        super().__init__()

        NewBieInitializer.init(
            model=self,
            model_config=model_config,
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int = 28,
        height: int = 1024,
        width: int = 1024,
        guidance: float = 5.0,
        init_image_path: str | None = None,
        init_image_strength: float = 0.3,
    ) -> Image.Image:
        """
        Generate an image from a text prompt.

        Args:
            seed: Random seed for reproducibility
            prompt: Text prompt describing the desired image
            num_inference_steps: Number of denoising steps (default 28)
            height: Output image height in pixels
            width: Output image width in pixels
            guidance: Classifier-free guidance scale (default 5.0)
            init_image_path: Optional path to initial image for img2img
            init_image_strength: Strength of initial image influence (0.0-1.0)

        Returns:
            Generated PIL Image
        """
        # 0. Create a new config based on the model type and input parameters
        from mflux.models.common.config import Config

        config = Config(
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance=guidance,
            image_path=init_image_path,
            image_strength=init_image_strength,
        )

        # 1. Initialize scheduler
        config._scheduler = FlowMatchEulerDiscreteScheduler(config=config)

        # 2. Encode text prompts
        text_embeddings = self._encode_prompt(prompt)

        # 3. Create latent shape
        latent_height = config.height // 8  # VAE downscale factor
        latent_width = config.width // 8
        latent_shape = (1, 16, latent_height, latent_width)  # 16-channel VAE

        # 4. Initialize random latents
        mx.random.seed(seed)
        latents = mx.random.normal(latent_shape)

        # 5. Handle init image if provided (img2img)
        if config.image_path is not None and config.image_strength is not None:
            latents = self._prepare_init_latents(
                init_image_path=str(config.image_path),
                latents=latents,
                strength=config.image_strength,
                scheduler=config.scheduler,
            )

        # 6. Create callback context and call before_loop
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)

        # 7. Denoising loop
        for t in config.time_steps:
            try:
                # Prepare timestep tensor
                timestep = mx.array([t])

                # Prepare guidance tensor
                guidance_tensor = mx.array([config.guidance])

                # Classifier-free guidance: predict noise for both conditional and unconditional
                if config.guidance > 1.0:
                    # Unconditional prediction (empty prompt)
                    uncond_embeddings = self._encode_prompt("")
                    noise_uncond = self.transformer(
                        latents=latents,
                        timestep=timestep,
                        text_embeddings=uncond_embeddings["gemma3"],
                        clip_embeddings=uncond_embeddings.get("jina_clip"),
                        guidance=guidance_tensor,
                    )

                    # Conditional prediction
                    noise_cond = self.transformer(
                        latents=latents,
                        timestep=timestep,
                        text_embeddings=text_embeddings["gemma3"],
                        clip_embeddings=text_embeddings.get("jina_clip"),
                        guidance=guidance_tensor,
                    )

                    # CFG combination
                    noise_pred = noise_uncond + config.guidance * (noise_cond - noise_uncond)
                else:
                    # No CFG, just conditional prediction
                    noise_pred = self.transformer(
                        latents=latents,
                        timestep=timestep,
                        text_embeddings=text_embeddings["gemma3"],
                        clip_embeddings=text_embeddings.get("jina_clip"),
                        guidance=guidance_tensor,
                    )

                # Scheduler step
                latents = config.scheduler.step(
                    model_output=noise_pred,
                    timestep=t,
                    sample=latents,
                )

                # Call subscribers in-loop
                ctx.in_loop(t, latents)

                # Evaluate for memory efficiency
                mx.eval(latents)

            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                from mflux.utils.exceptions import StopImageGenerationException
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )

        # 8. Call subscribers after loop
        ctx.after_loop(latents)

        # 9. Decode latents to image
        image = self._decode_latents(latents)

        return image

    def _encode_prompt(self, prompt: str) -> dict[str, mx.array]:
        """
        Encode text prompt using dual encoders.

        Args:
            prompt: Text prompt to encode

        Returns:
            Dictionary with 'gemma3' and 'jina_clip' embeddings
        """
        # Check cache
        if prompt in self.prompt_cache:
            return self.prompt_cache[prompt]

        embeddings = {}

        # Encode with Gemma3
        gemma3_tokens = self.tokenizers["gemma3"].encode(prompt)
        gemma3_tokens = mx.array(gemma3_tokens)[None, :]  # Add batch dim
        embeddings["gemma3"] = self.gemma3_text_encoder(gemma3_tokens)

        # Encode with Jina CLIP
        if "jina_clip" in self.tokenizers:
            clip_tokens = self.tokenizers["jina_clip"].encode(prompt)
            clip_tokens = mx.array(clip_tokens)[None, :]
            embeddings["jina_clip"] = self.jina_clip_encoder(clip_tokens)

        # Cache and return
        self.prompt_cache[prompt] = embeddings
        return embeddings

    def _decode_latents(self, latents: mx.array) -> Image.Image:
        """
        Decode latents to PIL Image.

        Args:
            latents: Latent tensor [1, 16, H, W]

        Returns:
            PIL Image
        """
        # Scale latents (VAE scaling factor)
        latents = latents / 0.18215

        # Decode through VAE
        decoded = self.vae.decode(latents)

        # Convert to numpy and then PIL
        decoded = mx.clip((decoded + 1) / 2, 0, 1)  # Rescale to [0, 1]
        decoded = (decoded * 255).astype(mx.uint8)
        decoded = decoded.squeeze(0).transpose(1, 2, 0)  # [H, W, C]

        # Convert to numpy
        image_array = decoded.__array__()
        image = Image.fromarray(image_array, mode="RGB")

        return image

    def _prepare_init_latents(
        self,
        init_image_path: str,
        latents: mx.array,
        strength: float,
        scheduler,
    ) -> mx.array:
        """
        Prepare latents from initial image for img2img.

        Args:
            init_image_path: Path to initial image
            latents: Noise latents
            strength: How much to transform the image (0.0 = no change, 1.0 = full noise)
            scheduler: Scheduler instance

        Returns:
            Prepared latents
        """
        # Load and preprocess image
        from PIL import Image
        import numpy as np

        image = Image.open(init_image_path).convert("RGB")

        # Resize to match latent dimensions
        target_h = latents.shape[2] * 8
        target_w = latents.shape[3] * 8
        image = image.resize((target_w, target_h), Image.Resampling.LANCZOS)

        # Convert to tensor
        image_array = np.array(image).astype(np.float32) / 255.0
        image_tensor = mx.array(image_array)
        image_tensor = image_tensor.transpose(2, 0, 1)[None, ...]  # [1, C, H, W]
        image_tensor = (image_tensor * 2) - 1  # Normalize to [-1, 1]

        # Encode through VAE
        init_latents = self.vae.encode(image_tensor) * 0.18215

        # Add noise based on strength
        noise_timestep = int(len(scheduler.timesteps) * strength)
        if noise_timestep > 0:
            noise = latents
            init_latents = scheduler.add_noise(
                init_latents,
                noise,
                scheduler.timesteps[noise_timestep - 1],
            )

        return init_latents

    def save_model(self, base_path: str) -> None:
        """
        Save the model to disk.

        Args:
            base_path: Directory path to save the model
        """
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=NewBieWeightDefinition,
        )

    def freeze(self, **kwargs):
        """Freeze all model parameters."""
        self.vae.freeze()
        self.transformer.freeze()
        self.gemma3_text_encoder.freeze()
        self.jina_clip_encoder.freeze()
