"""
Hunyuan-DiT Text-to-Image Model.

Main model class for Hunyuan-DiT, a DiT-based diffusion model with:
- 28 DiT transformer blocks with self-attention + cross-attention
- Dual text encoders: CLIP (1024 dim) + T5 (2048 dim)
- Standard 4-channel VAE
- DDPM scheduler with noise prediction
- Bilingual support (English + Chinese)
"""

from pathlib import Path

import mlx.core as mx
from mlx import nn

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.latent_creator.latent_creator import Img2Img, LatentCreator
from mflux.models.common.schedulers.ddpm_scheduler import DDPMScheduler
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
from mflux.models.flux.model.flux_text_encoder.clip_encoder.clip_encoder import CLIPEncoder
from mflux.models.flux.model.flux_text_encoder.t5_encoder.t5_encoder import T5Encoder
from mflux.models.flux.model.flux_vae.vae import VAE
from mflux.models.hunyuan.hunyuan_initializer import HunyuanInitializer
from mflux.models.hunyuan.model.hunyuan_transformer.hunyuan_dit import HunyuanDiT
from mflux.models.hunyuan.weights.hunyuan_weight_definition import HunyuanWeightDefinition
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil


class Hunyuan(nn.Module):
    """
    Hunyuan-DiT Text-to-Image Model.

    A DiT-based diffusion transformer with dual text encoders for
    bilingual (English + Chinese) text-to-image generation.

    Attributes:
        vae: VAE encoder/decoder (4-channel standard)
        transformer: HunyuanDiT with 28 blocks
        clip_text_encoder: Chinese CLIP encoder (1024 dim)
        t5_text_encoder: mT5-XXL encoder (2048 dim)
    """

    vae: VAE
    transformer: HunyuanDiT
    clip_text_encoder: CLIPEncoder
    t5_text_encoder: T5Encoder

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        model_config: ModelConfig = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ):
        """
        Initialize Hunyuan-DiT model.

        Args:
            quantize: Quantization bit width (4 or 8) or None for full precision
            model_path: Path to model weights (local or HuggingFace repo ID)
            model_config: Model configuration (defaults to Hunyuan config)
            lora_paths: List of LoRA paths (local files or HuggingFace repos)
            lora_scales: List of LoRA scale factors (default 1.0)
        """
        super().__init__()
        if model_config is None:
            model_config = ModelConfig.hunyuan()

        HunyuanInitializer.init(
            model=self,
            quantize=quantize,
            model_path=model_path,
            model_config=model_config,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int = 50,  # Hunyuan recommends 50 steps
        height: int = 1024,
        width: int = 1024,
        guidance: float = 7.5,  # Hunyuan default guidance
        image_path: Path | str | None = None,
        image_strength: float | None = None,
        negative_prompt: str | None = None,
    ) -> GeneratedImage:
        """
        Generate an image from a text prompt.

        Args:
            seed: Random seed for reproducibility
            prompt: Text description of the image to generate
            num_inference_steps: Number of denoising steps (default: 50)
            height: Image height in pixels (default: 1024)
            width: Image width in pixels (default: 1024)
            guidance: Guidance scale (default: 7.5)
            image_path: Optional input image for img2img
            image_strength: Strength for img2img (0.0-1.0)
            negative_prompt: Optional negative prompt

        Returns:
            GeneratedImage object containing the generated image and metadata
        """
        # 0. Create a new config with DDPM scheduler
        config = Config(
            width=width,
            height=height,
            guidance=guidance,
            scheduler="ddpm",
            image_path=image_path,
            image_strength=image_strength,
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
        )

        # Override scheduler with DDPMScheduler
        config.scheduler = DDPMScheduler(config=config)

        # 1. Create the initial latents
        latents = LatentCreator.create_for_txt2img_or_img2img(
            seed=seed,
            width=config.width,
            height=config.height,
            img2img=Img2Img(
                vae=self.vae,
                latent_creator=FluxLatentCreator,
                image_path=config.image_path,
                sigmas=config.scheduler.sigmas,
                init_time_step=config.init_time_step,
            ),
        )

        # 2. Encode the prompt (both CLIP and T5)
        clip_embeds, t5_embeds = self._encode_prompt(prompt)

        # Encode negative prompt if provided, or create empty embeddings for CFG
        if config.guidance > 1.0:
            if negative_prompt:
                neg_clip_embeds, neg_t5_embeds = self._encode_prompt(negative_prompt)
            else:
                # Use empty embeddings for unconditional guidance
                neg_clip_embeds = mx.zeros_like(clip_embeds)
                neg_t5_embeds = mx.zeros_like(t5_embeds)
        else:
            neg_clip_embeds, neg_t5_embeds = None, None

        # 3. Create callback context and call before_loop
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)

        for t in range(config.num_inference_steps):
            try:
                # Scale model input if needed by the scheduler
                latents_input = config.scheduler.scale_model_input(latents, t)

                # 4.t Predict the noise (conditional)
                noise_pred = self.transformer(
                    t=t,
                    config=config,
                    hidden_states=latents_input,
                    clip_embeds=clip_embeds,
                    t5_embeds=t5_embeds,
                )

                # Classifier-free guidance
                if config.guidance > 1.0 and neg_clip_embeds is not None:
                    # Predict unconditional noise
                    noise_pred_uncond = self.transformer(
                        t=t,
                        config=config,
                        hidden_states=latents_input,
                        clip_embeds=neg_clip_embeds,
                        t5_embeds=neg_t5_embeds,
                    )
                    # Apply guidance
                    noise_pred = noise_pred_uncond + config.guidance * (noise_pred - noise_pred_uncond)

                # 5.t Take one denoise step
                latents = config.scheduler.step(noise=noise_pred, timestep=t, latents=latents)

                # 6.t Call subscribers in-loop
                ctx.in_loop(t, latents)

                # (Optional) Evaluate to enable progress tracking
                mx.eval(latents)

            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )

        # 7. Call subscribers after loop
        ctx.after_loop(latents)

        # 8. Decode the latent array and return the image
        # Hunyuan uses standard 4-channel VAE, no unpacking needed
        decoded = VAEUtil.decode(vae=self.vae, latent=latents, tiling_config=self.tiling_config)
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
            generation_time=ctx.elapsed_time if hasattr(ctx, 'elapsed_time') else 0,
        )

    def _encode_prompt(self, prompt: str) -> tuple[mx.array, mx.array]:
        """
        Encode a text prompt using both CLIP and T5.

        Args:
            prompt: Text to encode

        Returns:
            Tuple of (clip_embeds [batch, 77, 1024], t5_embeds [batch, 256, 2048])
        """
        # Check cache first
        if prompt in self.prompt_cache:
            return self.prompt_cache[prompt]

        # Tokenize and encode with CLIP
        clip_tokenizer = self.tokenizers["clip"]
        clip_output = clip_tokenizer.tokenize(prompt)
        clip_embeds = self.clip_text_encoder(clip_output.input_ids)

        # Tokenize and encode with T5
        t5_tokenizer = self.tokenizers["t5"]
        t5_output = t5_tokenizer.tokenize(prompt)
        t5_embeds = self.t5_text_encoder(t5_output.input_ids)

        # Cache and return
        self.prompt_cache[prompt] = (clip_embeds, t5_embeds)
        return clip_embeds, t5_embeds

    @staticmethod
    def from_name(model_name: str, quantize: int | None = None) -> "Hunyuan":
        """
        Create a Hunyuan model from a model name.

        Args:
            model_name: Model name or alias (e.g., "hunyuan", "hunyuan-dit")
            quantize: Quantization bit width (4 or 8) or None

        Returns:
            Initialized Hunyuan model
        """
        return Hunyuan(
            model_config=ModelConfig.from_name(model_name=model_name, base_model=None),
            quantize=quantize,
        )

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
            weight_definition=HunyuanWeightDefinition,
        )

    def freeze(self, **kwargs):
        """Freeze all model parameters."""
        self.vae.freeze()
        self.transformer.freeze()
        self.clip_text_encoder.freeze()
        self.t5_text_encoder.freeze()
