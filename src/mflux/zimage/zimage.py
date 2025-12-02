"""Z-Image-Turbo model for text-to-image generation.

Main entry point for generating images using Z-Image-Turbo.
Handles model loading, text encoding, diffusion, and VAE decoding.
"""

import logging

import mlx.core as mx
from mlx import nn
from tqdm import tqdm

from mflux.callbacks.callbacks import Callbacks
from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.config.runtime_config import RuntimeConfig
from mflux.models.common.latent_creator.latent_creator import Img2Img, LatentCreator
from mflux.models.common.weights.model_saver import ModelSaver
from mflux.models.flux.model.flux_vae.vae import VAE
from mflux.models.zimage.latent_creator.zimage_latent_creator import ZImageLatentCreator
from mflux.models.zimage.scheduler import ZImageScheduler
from mflux.models.zimage.text_encoder import Qwen3Encoder, Qwen3Tokenizer
from mflux.models.zimage.transformer import S3DiT
from mflux.models.zimage.weights import ZImageWeightHandler
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil

log = logging.getLogger(__name__)


class ZImage(nn.Module):
    """Z-Image-Turbo model for text-to-image generation.

    Main entry point for generating images using Z-Image-Turbo.
    Handles model loading, text encoding, diffusion, and VAE decoding.
    """

    # Architecture constants
    VAE_SCALE_FACTOR = 8  # VAE spatial downsampling factor
    LATENT_CHANNELS = 16  # VAE latent channels

    # Default inference settings (from research)
    DEFAULT_STEPS = 9  # Turbo uses 9 steps (8 + 1)
    DEFAULT_GUIDANCE_SCALE = 0.0  # CFG baked into distilled weights

    # Supported model variants
    VARIANTS = {
        "turbo": "Tongyi-MAI/Z-Image-Turbo",
    }

    vae: VAE
    transformer: S3DiT
    text_encoder: Qwen3Encoder
    tokenizer: Qwen3Tokenizer

    def __init__(
        self,
        model_config: ModelConfig,
        quantize: int | None = None,
        local_path: str | None = None,
    ):
        """Initialize ZImage model.

        Args:
            model_config: Model configuration
            quantize: Quantization bits (3, 4, 5, 6, 8) or None for fp16
            local_path: Optional local path to model weights
        """
        super().__init__()
        self.model_config = model_config
        self.bits = quantize
        self.local_path = local_path

        # Load all components
        self._init_components(quantize, local_path)

    def _init_components(
        self,
        quantize: int | None,
        local_path: str | None,
    ) -> None:
        """Initialize all model components.

        Args:
            quantize: Quantization bits
            local_path: Optional local path
        """
        # Load weights and create models
        components = ZImageWeightHandler.load_model(
            alias=self.model_config.aliases[0],
            quantize=quantize,
            local_path=local_path,
        )

        self.transformer = components.transformer
        self.text_encoder = components.text_encoder
        self.tokenizer = components.tokenizer
        self.vae = components.vae

        # Fix self.bits for pre-quantized models loaded via local_path
        # When loading from local_path, self.bits is None even though model IS quantized
        if self.bits is None:
            self.bits = self._detect_quantization_bits()

    def _detect_quantization_bits(self) -> int | None:
        """Detect quantization bits from loaded model layers.

        Returns:
            Quantization bits (3, 4, 5, 6, 8) or None if not quantized
        """
        for module in self.transformer.modules():
            if isinstance(module, nn.QuantizedLinear):
                return module.bits
        return None

    def generate_image(
        self,
        seed: int,
        prompt: str,
        config: Config | None = None,
    ) -> GeneratedImage:
        """Generate image from text prompt (txt2img or img2img).

        Args:
            seed: Random seed for reproducibility
            prompt: Text description of desired image
            config: Generation config (steps, height, width, image_path, image_strength)

        Returns:
            GeneratedImage with PIL Image and metadata
        """
        # Use default config if not provided
        if config is None:
            config = Config(
                num_inference_steps=self.DEFAULT_STEPS,
                height=1024,
                width=1024,
                guidance=self.DEFAULT_GUIDANCE_SCALE,
            )

        # Create runtime config
        runtime_config = RuntimeConfig(config, self.model_config)

        # Calculate start step based on img2img strength
        init_time_step = runtime_config.init_time_step
        time_steps = tqdm(range(init_time_step, runtime_config.num_inference_steps))

        # Memory tracking - start of generation
        log.debug(f"Memory at generation start: {mx.get_active_memory() / 1e9:.2f} GB")

        # 1. Create scheduler with shift=3.0 for Z-Image
        scheduler = ZImageScheduler(runtime_config)

        # 2. Encode text prompt
        text_embeddings = self._encode_prompt(prompt)

        # Memory tracking - after text encoding
        log.debug(f"Memory after text encoding: {mx.get_active_memory() / 1e9:.2f} GB")

        # 3. Create initial latents (txt2img or img2img)
        latents = LatentCreator.create_for_txt2img_or_img2img(
            seed=seed,
            height=runtime_config.height,
            width=runtime_config.width,
            img2img=Img2Img(
                vae=self.vae,
                latent_creator=ZImageLatentCreator,
                image_path=runtime_config.image_path,
                sigmas=scheduler.sigmas,
                init_time_step=init_time_step,
            ),
        )

        # (Optional) Call subscribers for beginning of loop
        Callbacks.before_loop(
            seed=seed,
            prompt=prompt,
            latents=latents,
            config=runtime_config,
        )

        # Precompute RoPE frequencies (constant for all steps)
        rope = self.transformer.precompute_rope(
            runtime_config.height // self.VAE_SCALE_FACTOR,
            runtime_config.width // self.VAE_SCALE_FACTOR,
            text_embeddings.shape[1],
        )
        mx.eval(rope[0], rope[1])  # Force evaluation

        # Precompute timestep embeddings (constant for all steps)
        all_temb = self.transformer.precompute_timestep_embeddings(scheduler, runtime_config.num_inference_steps)
        mx.eval(all_temb)

        # 4. Denoising loop (starts at init_time_step for img2img, 0 for txt2img)
        for t_idx in time_steps:
            try:
                # Get velocity prediction from transformer
                velocity = self.transformer(
                    latents=latents,
                    text_embeddings=text_embeddings,
                    timestep=None,  # Not needed when temb provided
                    rope=rope,  # Pass precomputed RoPE frequencies
                    temb=all_temb[t_idx : t_idx + 1],  # Select embedding for current step
                )

                # Unpatchify to spatial format for Euler step
                velocity = self.transformer.unpatchify(
                    velocity,
                    height=runtime_config.height // self.VAE_SCALE_FACTOR,
                    width=runtime_config.width // self.VAE_SCALE_FACTOR,
                )

                # CRITICAL: Negate velocity (diffusers pipeline_z_image.py line 560)
                # The transformer predicts velocity in the opposite direction
                velocity = -velocity

                # Euler step using scheduler
                latents = scheduler.step(velocity, t_idx, latents)

                # (Optional) Call subscribers in-loop
                Callbacks.in_loop(
                    t=t_idx,
                    seed=seed,
                    prompt=prompt,
                    latents=latents,
                    config=runtime_config,
                    time_steps=time_steps,
                )

                # Evaluate to enable progress tracking
                mx.eval(latents)

            except KeyboardInterrupt:  # noqa: PERF203
                Callbacks.interruption(
                    t=t_idx,
                    seed=seed,
                    prompt=prompt,
                    latents=latents,
                    config=runtime_config,
                    time_steps=time_steps,
                )
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t_idx + 1}/{runtime_config.num_inference_steps}"
                )

        # (Optional) Call subscribers after loop
        Callbacks.after_loop(
            seed=seed,
            prompt=prompt,
            latents=latents,
            config=runtime_config,
        )

        # Clean up precomputed tensors that are no longer needed
        # This prevents them from being held in memory during VAE decode
        del rope
        del all_temb

        # Memory tracking - after diffusion loop
        log.debug(f"Memory after diffusion loop: {mx.get_active_memory() / 1e9:.2f} GB")

        # 5. Decode latents with VAE
        decoded = self.vae.decode(latents)

        # Memory tracking - after VAE decode (before eval)
        log.debug(f"Memory after VAE decode (unevaluated): {mx.get_active_memory() / 1e9:.2f} GB")

        # CRITICAL: Evaluate decoded latents to prevent graph explosion
        # Without this, the computation graph grows to 30+GB during image conversion
        mx.eval(decoded)

        # Memory tracking - after VAE decode eval
        log.debug(f"Memory after VAE decode (evaluated): {mx.get_active_memory() / 1e9:.2f} GB")

        # 6. Convert to GeneratedImage
        result = ImageUtil.to_image(
            decoded_latents=decoded,
            config=runtime_config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            generation_time=time_steps.format_dict["elapsed"],
            lora_paths=None,
            lora_scales=None,
            image_path=runtime_config.image_path,
            image_strength=runtime_config.image_strength,
        )

        # Memory tracking - after image conversion
        log.debug(f"Memory after image conversion: {mx.get_active_memory() / 1e9:.2f} GB")

        # Clean up intermediate tensors to prevent graph retention during exit
        # The decoded tensor is no longer needed after image conversion
        del decoded
        del latents
        del text_embeddings

        # Clear MLX cache to release unused memory before returning
        mx.clear_cache()

        return result

    def _encode_prompt(self, prompt: str, max_sequence_length: int = 512) -> mx.array:
        """Encode text prompt to embeddings.

        Matches diffusers ZImagePipeline.encode_prompt behavior:
        - Applies chat template before tokenization
        - Uses hidden_states[-2] from text encoder
        - Extracts only non-padding tokens

        Args:
            prompt: Text prompt
            max_sequence_length: Maximum sequence length (default 512 like diffusers)

        Returns:
            Text embeddings [1, seq_len, 2560] (batch dim kept for transformer)
        """
        # Tokenize (with chat template applied)
        tokens = self.tokenizer(prompt, max_length=max_sequence_length)
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        # Encode with Qwen3 (returns hidden_states[-2] by default)
        embeddings = self.text_encoder(input_ids, attention_mask)

        # Extract only non-padding tokens (like diffusers)
        # attention_mask: [B, seq_len], 1=valid, 0=pad
        # embeddings: [B, seq_len, 2560]
        # Extract first batch item
        emb = embeddings[0]  # [seq_len, 2560]
        mask = attention_mask[0]  # [seq_len]

        # MLX doesn't support boolean indexing, so we count valid tokens and slice
        # NOTE: .item() here is acceptable - this is called ONCE per generation (not in hot path)
        # and we need the actual integer value for slicing. Impact: <1ms vs 9+ calls in loop.
        valid_len = int(mx.sum(mask).item())
        emb = emb[:valid_len, :]  # [valid_len, 2560]

        # Add batch dimension back for transformer
        emb = emb[None, :, :]  # [1, valid_len, 2560]

        return emb

    def _init_latents(
        self,
        seed: int,
        height: int,
        width: int,
    ) -> mx.array:
        """Initialize random latents for diffusion.

        Args:
            seed: Random seed
            height: Image height (pixels)
            width: Image width (pixels)

        Returns:
            Random latents [1, C, H/8, W/8]
        """
        mx.random.seed(seed)

        # VAE produces 8x downsampled latents
        latent_h = height // self.VAE_SCALE_FACTOR
        latent_w = width // self.VAE_SCALE_FACTOR

        # 16 channels for VAE latents, BCHW format
        latents = mx.random.normal((1, self.LATENT_CHANNELS, latent_h, latent_w))

        return latents

    def save_model(self, base_path: str) -> None:
        """Save model weights and tokenizer to disk.

        Saves pre-quantized weights that can be loaded without peak memory spike.
        Following Qwen pattern: only transformer is quantized, text encoder stays fp16
        to avoid semantic degradation.

        Args:
            base_path: Directory to save model files
        """
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
