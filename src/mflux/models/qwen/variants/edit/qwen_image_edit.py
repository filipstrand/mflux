import math

import mlx.core as mx
from mlx import nn
from PIL import Image
from tqdm import tqdm

from mflux.callbacks.callbacks import Callbacks
from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.config.runtime_config import RuntimeConfig
from mflux.error.exceptions import StopImageGenerationException
from mflux.latent_creator.latent_creator import LatentCreator
from mflux.models.qwen.model.qwen_text_encoder.qwen_text_encoder import QwenTextEncoder
from mflux.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
from mflux.models.qwen.qwen_edit_initializer import QwenImageEditInitializer
from mflux.models.qwen.variants.edit.utils.qwen_edit_util import QwenEditUtil
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from mflux.post_processing.array_util import ArrayUtil
from mflux.post_processing.generated_image import GeneratedImage
from mflux.post_processing.image_util import ImageUtil


class QwenImageEdit(nn.Module):
    vae: QwenVAE
    transformer: QwenTransformer
    text_encoder: QwenTextEncoder

    def __init__(
        self,
        quantize: int | None = None,
        local_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        lora_names: list[str] | None = None,
        lora_repo_id: str | None = None,
    ):
        super().__init__()
        QwenImageEditInitializer.init(
            qwen_model=self,
            model_config=ModelConfig.qwen_image_edit(),
            quantize=quantize,
            local_path=local_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            lora_names=lora_names,
            lora_repo_id=lora_repo_id,
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        config: Config,
        negative_prompt: str | None = None,
    ) -> GeneratedImage:
        image = ImageUtil.load_image(config.image_path).convert("RGB")
        image_size = image.size  # (width, height)

        # Use Diffusers' calculate_dimensions function logic (lines 155-162)
        target_area = 1024 * 1024
        ratio = image_size[0] / image_size[1]  # width / height
        calculated_width = math.sqrt(target_area * ratio)
        calculated_height = calculated_width / ratio
        calculated_width = round(calculated_width / 32) * 32
        calculated_height = round(calculated_height / 32) * 32

        # Use calculated dimensions or provided ones for OUTPUT (like Diffusers lines 671-672)
        use_height = config.height or calculated_height
        use_width = config.width or calculated_width

        # Apply VAE scale factor constraint (like Diffusers lines 674-676)
        vae_scale_factor = 8  # Same as Diffusers
        multiple_of = vae_scale_factor * 2  # 16
        use_width = use_width // multiple_of * multiple_of
        use_height = use_height // multiple_of * multiple_of

        # Create config for OUTPUT dimensions
        final_config = Config(
            height=use_height,
            width=use_width,
            image_path=config.image_path,
            num_inference_steps=config.num_inference_steps,
            guidance=config.guidance,
            scheduler=config.scheduler_str,  # Preserve scheduler choice
        )
        runtime_config = RuntimeConfig(final_config, self.model_config)

        # NOT the output dimensions! This ensures conditioning matches PyTorch
        # The regular edit model uses calculated dimensions (from 1024x1024 target) for both
        # vision encoder and VAE encoding, unlike Edit Plus which uses 384x384 for vision encoder
        vl_width = calculated_width
        vl_height = calculated_height

        # Get timesteps from the scheduler (these are the actual time values, not just indices)
        timesteps = runtime_config.scheduler.timesteps
        time_steps = tqdm(enumerate(timesteps), total=len(timesteps))

        # 1. Create initial latents
        latents = LatentCreator.create(
            seed=seed,
            height=runtime_config.height,
            width=runtime_config.width,
        )

        # 2. Generate prompt embeddings with MLX encoder
        prompt_embeds, prompt_mask, negative_prompt_embeds, negative_prompt_mask = (
            self._debug_encode_prompts_with_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image_path=runtime_config.image_path,
                runtime_config=runtime_config,
                vl_width=int(vl_width),
                vl_height=int(vl_height),
            )
        )
        # Note: We're using MLX text encoder output (not loading from PyTorch)

        # 3. Generate image conditioning latents with MLX (compare with diffusers)
        # The function uses vl_width/vl_height for encoding (ignores height/width params)
        # So we pass calculated dimensions which will be used for both vision encoder AND VAE encoding
        # This ensures consistency - both see the same resolution (calculated from 1024x1024 target)
        # Convert Path to string if needed (function accepts list or str)
        image_path = str(runtime_config.image_path) if runtime_config.image_path else None
        static_image_latents, qwen_image_ids, cond_h_patches, cond_w_patches, num_images = (
            QwenEditUtil.create_image_conditioning_latents(
                vae=self.vae,
                height=runtime_config.height,  # Ignored when vl_width/vl_height provided
                width=runtime_config.width,  # Ignored when vl_width/vl_height provided
                image_paths=image_path,
                vl_width=int(vl_width),  # Calculated dimensions used for encoding (matches vision encoder)
                vl_height=int(vl_height),  # This ensures vision encoder and VAE encoding match
            )
        )
        # The function uses vl_width/vl_height to compute calc_h/calc_w, which determines patch counts

        # (Optional) Call subscribers for beginning of loop
        Callbacks.before_loop(
            seed=seed,
            prompt=prompt,
            latents=latents,
            config=runtime_config,
        )

        for step_idx, timestep_value in time_steps:
            try:
                # 4.t Concatenate the updated latents with the static image latents
                hidden_states = mx.concatenate([latents, static_image_latents], axis=1)
                hidden_states_neg = mx.concatenate([latents, static_image_latents], axis=1)  # noqa: F841

                # 5.t Predict the noise
                # Note: Pass the sigma value (timestep / 1000) to the transformer, matching PyTorch

                # 🔧 Use transformer with MLX's native RoPE computation
                noise = self.transformer(
                    t=float(timestep_value.item()) / 1000.0,  # Convert to sigma [0, 1]
                    config=runtime_config,
                    hidden_states=hidden_states,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_mask,
                    qwen_image_ids=qwen_image_ids,  # Pass image IDs for conditioning
                    cond_image_grid=(1, cond_h_patches, cond_w_patches),  # Use VAE conditioning grid, not VL grid
                )[:, : latents.shape[1]]

                noise_negative = self.transformer(
                    t=float(timestep_value.item()) / 1000.0,  # Convert to sigma [0, 1]
                    config=runtime_config,
                    hidden_states=hidden_states_neg,
                    encoder_hidden_states=negative_prompt_embeds,
                    encoder_hidden_states_mask=negative_prompt_mask,
                    qwen_image_ids=qwen_image_ids,
                    cond_image_grid=(1, cond_h_patches, cond_w_patches),  # Use VAE conditioning grid, not VL grid
                )[:, : latents.shape[1]]

                guided_noise = QwenImage.compute_guided_noise(noise, noise_negative, runtime_config.guidance)

                # 6.t Take one denoise step
                # Note: Pass the step index (0, 1, 2...) to the scheduler, not the timestep value
                latents = runtime_config.scheduler.step(
                    model_output=guided_noise,
                    timestep=step_idx,  # Scheduler uses index
                    sample=latents,
                )

                # (Optional) Call subscribers in-loop
                Callbacks.in_loop(
                    t=step_idx,
                    seed=seed,
                    prompt=prompt,
                    latents=latents,
                    config=runtime_config,
                    time_steps=time_steps,
                )

                # (Optional) Evaluate to enable progress tracking
                mx.eval(latents)

            except KeyboardInterrupt:  # noqa: PERF203
                Callbacks.interruption(
                    t=step_idx,
                    seed=seed,
                    prompt=prompt,
                    latents=latents,
                    config=runtime_config,
                    time_steps=time_steps,
                )
                raise StopImageGenerationException(f"Stopping image generation at step {step_idx + 1}/{len(timesteps)}")

        # (Optional) Call subscribers after loop
        Callbacks.after_loop(
            seed=seed,
            prompt=prompt,
            latents=latents,
            config=runtime_config,
        )

        # 7. Decode the latent array and return the image
        latents = ArrayUtil.unpack_latents(latents=latents, height=runtime_config.height, width=runtime_config.width)
        decoded = self.vae.decode(latents)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=runtime_config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            image_path=runtime_config.image_path,
            generation_time=time_steps.format_dict["elapsed"],
            negative_prompt=negative_prompt,
        )

    def _debug_encode_prompts_with_image(
        self,
        prompt: str,
        negative_prompt: str,
        image_path: str,
        runtime_config,
        vl_width: int | None = None,
        vl_height: int | None = None,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        # 1. Use existing tokenizer and load image
        tokenizer = self.qwen_vl_tokenizer  # Use the already initialized tokenizer
        image = Image.open(image_path).convert("RGB")

        # 2. Tokenize positive prompt (tokenizer will apply template)
        pos_input_ids, pos_attention_mask, pos_pixel_values, pos_image_grid_thw = tokenizer.tokenize_with_image(
            prompt, image, vl_width=vl_width, vl_height=vl_height
        )

        # 3. Run text encoder on positive prompt
        pos_hidden_states = self.qwen_vl_encoder(
            input_ids=pos_input_ids,
            attention_mask=pos_attention_mask,
            pixel_values=pos_pixel_values,
            image_grid_thw=pos_image_grid_thw,
        )

        # 4. Tokenize and encode negative prompt (tokenizer will apply template)
        # Use empty string as default negative prompt if None is provided
        neg_prompt = negative_prompt if negative_prompt is not None else ""
        neg_input_ids, neg_attention_mask, neg_pixel_values, neg_image_grid_thw = tokenizer.tokenize_with_image(
            neg_prompt, image, vl_width=vl_width, vl_height=vl_height
        )

        neg_hidden_states = self.qwen_vl_encoder(
            input_ids=neg_input_ids,
            attention_mask=neg_attention_mask,
            pixel_values=neg_pixel_values,
            image_grid_thw=neg_image_grid_thw,
        )

        # Return the embeddings for use in the pipeline
        return (
            pos_hidden_states[0].astype(mx.float16),  # prompt_embeds
            pos_hidden_states[1].astype(mx.float16),  # prompt_mask
            neg_hidden_states[0].astype(mx.float16),  # negative_prompt_embeds
            neg_hidden_states[1].astype(mx.float16),  # negative_prompt_mask
        )
