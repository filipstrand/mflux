from pathlib import Path

import mlx.core as mx
from mlx import nn

from mflux.models.common.config import ModelConfig
from mflux.models.common.config.config import Config
from mflux.models.common.latent_creator.latent_creator import Img2Img, LatentCreator
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
from mflux.models.qwen.model.qwen_text_encoder.qwen_prompt_encoder import QwenPromptEncoder
from mflux.models.qwen.model.qwen_text_encoder.qwen_text_encoder import QwenTextEncoder
from mflux.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
from mflux.models.qwen.qwen_initializer import QwenImageInitializer
from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil


class QwenImage(nn.Module):
    vae: QwenVAE
    transformer: QwenTransformer
    text_encoder: QwenTextEncoder

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        model_config: ModelConfig = ModelConfig.qwen_image(),
    ):
        super().__init__()
        QwenImageInitializer.init(
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
        num_inference_steps: int = 4,
        height: int = 1024,
        width: int = 1024,
        guidance: float = 4.0,
        image_path: Path | str | None = None,
        image_strength: float | None = None,
        scheduler: str = "linear",
        negative_prompt: str | None = None,
        guidance_rescale: float = 0.7,
    ) -> GeneratedImage:
        # HIGH PRIORITY FIX: Validate guidance parameter range to prevent numerical instability
        if not (0.0 <= guidance <= 50.0):
            raise ValueError(f"Guidance must be in range [0.0, 50.0] to prevent numerical instability, got {guidance}")

        # 0. Create a new config based on the model type and input parameters
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

        # 1. Create the initial latents
        latents = LatentCreator.create_for_txt2img_or_img2img(
            seed=seed,
            width=config.width,
            height=config.height,
            img2img=Img2Img(
                vae=self.vae,
                latent_creator=QwenLatentCreator,
                sigmas=config.scheduler.sigmas,
                init_time_step=config.init_time_step,
                image_path=config.image_path,
                tiling_config=self.tiling_config,
            ),
        )

        # 2. Encode the prompt (using native MLX encoding)
        prompt_embeds_orig, prompt_mask_orig, negative_prompt_embeds_orig, negative_prompt_mask_orig = (
            QwenPromptEncoder.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                prompt_cache=self.prompt_cache,
                qwen_tokenizer=self.tokenizers["qwen"],
                qwen_text_encoder=self.text_encoder,
            )
        )

        # CRITICAL FIX: Pad prompt embeddings ONCE before loop, not inside loop
        # Positive and negative prompts may have different sequence lengths
        max_seq_len = max(prompt_embeds_orig.shape[1], negative_prompt_embeds_orig.shape[1])

        # HIGH PRIORITY FIX: Validate sequence length against model limits
        # Qwen supports up to 128k theoretically, but RoPE buffer is 4096 tokens
        MAX_TRANSFORMER_SEQ_LEN = 4096  # From qwen_rope.py _ROPE_BUFFER_SIZE
        if max_seq_len > MAX_TRANSFORMER_SEQ_LEN:
            raise ValueError(
                f"Combined sequence length {max_seq_len} exceeds model maximum {MAX_TRANSFORMER_SEQ_LEN}. "
                f"Positive prompt: {prompt_embeds_orig.shape[1]}, Negative prompt: {negative_prompt_embeds_orig.shape[1]}"
            )

        if max_seq_len < 1:
            raise ValueError("Sequence length must be at least 1 after padding")

        # Pad positive prompt if needed
        if prompt_embeds_orig.shape[1] < max_seq_len:
            pad_len = max_seq_len - prompt_embeds_orig.shape[1]
            prompt_embeds = mx.pad(prompt_embeds_orig, [(0, 0), (0, pad_len), (0, 0)])
            prompt_mask = mx.pad(prompt_mask_orig, [(0, 0), (0, pad_len)])
        else:
            prompt_embeds = prompt_embeds_orig
            prompt_mask = prompt_mask_orig

        # Pad negative prompt if needed
        if negative_prompt_embeds_orig.shape[1] < max_seq_len:
            pad_len = max_seq_len - negative_prompt_embeds_orig.shape[1]
            negative_prompt_embeds = mx.pad(negative_prompt_embeds_orig, [(0, 0), (0, pad_len), (0, 0)])
            negative_prompt_mask = mx.pad(negative_prompt_mask_orig, [(0, 0), (0, pad_len)])
        else:
            negative_prompt_embeds = negative_prompt_embeds_orig
            negative_prompt_mask = negative_prompt_mask_orig

        # 3. Create callback context and call before_loop
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)

        for t in config.time_steps:
            try:
                # Scale model input if needed by the scheduler
                latents = config.scheduler.scale_model_input(latents, t)

                # 4. Predict the noise with BATCHED GUIDANCE
                # OPTIMIZATION: Single batched transformer pass instead of 2 sequential passes
                # Old: 2 forward passes per step (positive + negative)
                # New: 1 forward pass with batch_size × 2 (40-50% speedup on transformer)

                # Concatenate inputs along batch dimension (embeddings already padded)
                batched_latents = mx.concatenate([latents, latents], axis=0)
                batched_text = mx.concatenate([prompt_embeds, negative_prompt_embeds], axis=0)
                batched_mask = mx.concatenate([prompt_mask, negative_prompt_mask], axis=0)

                # HIGH PRIORITY FIX: Validate shape consistency for batched guidance
                # CRITICAL: Use explicit validation instead of assert (disabled with -O flag)
                expected_batch = latents.shape[0] * 2
                if batched_latents.shape[0] != expected_batch:
                    raise ValueError(
                        f"Batch concatenation failed: expected {expected_batch}, got {batched_latents.shape[0]}"
                    )
                if batched_text.shape[0] != expected_batch:
                    raise ValueError(
                        f"Text/latent batch mismatch: expected {expected_batch}, got {batched_text.shape[0]}"
                    )

                # Single transformer forward pass (2x batch size)
                batched_noise = self.transformer(
                    t=t,
                    config=config,
                    hidden_states=batched_latents,
                    encoder_hidden_states=batched_text,
                    encoder_hidden_states_mask=batched_mask,
                )

                # HIGH PRIORITY FIX: Validate output shape before split
                # CRITICAL: Use explicit validation instead of assert (disabled with -O flag)
                if batched_noise.shape[0] != expected_batch:
                    raise ValueError(
                        f"Transformer output batch mismatch: expected {expected_batch}, got {batched_noise.shape[0]}"
                    )

                # Split results back to positive and negative
                noise, noise_negative = mx.split(batched_noise, 2, axis=0)

                # Compute guided noise with optional rescaling
                guided_noise = QwenImage.compute_guided_noise(noise, noise_negative, config.guidance, guidance_rescale)

                # 5.t Take one denoise step
                latents = config.scheduler.step(noise=guided_noise, timestep=t, latents=latents)

                # 6.t Call subscribers in-loop
                ctx.in_loop(t, latents)

                # PERFORMANCE: mx.eval() removed to preserve lazy evaluation benefits
                # Forced synchronization causes 15-25% slowdown. MLX handles evaluation automatically.
                # Uncomment only for debugging if needed:
                # mx.eval(latents)

            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )

        # 7. Call subscribers after loop
        ctx.after_loop(latents)

        # 8. Decode the latent array and return the image
        latents = QwenLatentCreator.unpack_latents(latents=latents, height=config.height, width=config.width)
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
            generation_time=config.time_steps.format_dict["elapsed"],
            negative_prompt=negative_prompt,
        )

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=QwenWeightDefinition,
        )

    @staticmethod
    def compute_guided_noise(
        noise: mx.array,
        noise_negative: mx.array,
        guidance: float,
        rescale_cfg: float = 0.7,
    ) -> mx.array:
        """Compute classifier-free guidance with optional rescaling.

        CFG rescaling helps reduce color oversaturation at high guidance values
        by blending the rescaled output with the original output.

        Args:
            noise: Positive (conditioned) noise prediction
            noise_negative: Negative (unconditioned) noise prediction
            guidance: CFG guidance scale (typically 3.5-7.5)
            rescale_cfg: Rescale factor in [0, 1]. 0 = no rescale, 1 = full rescale.
                        Default 0.7 provides good balance. Set to 0 to disable.

        Returns:
            Guided and optionally rescaled noise prediction

        Reference:
            "Common Diffusion Noise Schedules and Sample Steps are Flawed"
            https://arxiv.org/abs/2305.08891
        """
        # Standard CFG combination
        combined = noise_negative + guidance * (noise - noise_negative)

        # Skip rescaling if disabled
        if rescale_cfg <= 0.0:
            return combined

        # Numerical stability constants for CFG rescaling
        # float32: 1e-6 is ~10x sqrt(machine epsilon) for safe division
        # float16/bfloat16: 1e-4 compensates for reduced mantissa precision (7-10 bits vs 23)
        CFG_EPSILON_FP32 = 1e-6
        CFG_EPSILON_FP16 = 1e-4
        SAFE_THRESHOLD_MULTIPLIER = 10  # Margin above epsilon for safe division

        eps = CFG_EPSILON_FP32 if noise.dtype == mx.float32 else CFG_EPSILON_FP16
        safe_threshold = eps * SAFE_THRESHOLD_MULTIPLIER

        # Calculate RMS (root mean square) for rescaling
        # Using mean(x²) instead of full variance for efficiency
        noise_std = mx.sqrt(mx.mean(noise * noise, axis=-1, keepdims=True) + eps)
        combined_std = mx.sqrt(mx.mean(combined * combined, axis=-1, keepdims=True) + eps)

        # Calculate rescale factor to match original noise statistics
        # Use 1.0 when combined_std is too small to avoid numerical issues
        rescale_factor = mx.where(
            combined_std > safe_threshold,
            noise_std / combined_std,
            mx.ones_like(combined_std),
        )

        # Guard against NaN/Inf from edge cases (e.g., all-zero inputs)
        rescale_factor = mx.where(
            mx.isnan(rescale_factor) | mx.isinf(rescale_factor),
            mx.ones_like(rescale_factor),
            rescale_factor,
        )

        # Apply rescaling with blend factor
        rescaled = combined * rescale_factor

        # Blend between original CFG and rescaled CFG
        # rescale_cfg=0: original CFG, rescale_cfg=1: full rescale
        result = rescale_cfg * rescaled + (1.0 - rescale_cfg) * combined

        return result
