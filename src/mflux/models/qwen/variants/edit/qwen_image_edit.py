import math
import os
import numpy as np
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
from mflux.models.qwen.model.qwen_text_encoder.qwen_vision_language_prompt_encoder import (
    QwenVisionLanguagePromptEncoder,
)
from mflux.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
from mflux.models.qwen.qwen_edit_initializer import QwenImageEditInitializer
from mflux.models.qwen.variants.edit.utils.qwen_edit_util import QwenEditUtil
from mflux.models.qwen.variants.edit.utils.qwen_prompt_rewriter import QwenPromptRewriter
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

        # Use calculated dimensions or provided ones (like Diffusers lines 823-824)
        use_height = config.height or calculated_height
        use_width = config.width or calculated_width

        # Apply VAE scale factor constraint (like Diffusers lines 826-828)
        vae_scale_factor = 8  # Same as Diffusers
        multiple_of = vae_scale_factor * 2  # 16
        use_width = use_width // multiple_of * multiple_of
        use_height = use_height // multiple_of * multiple_of

        # Use chosen dimensions for both output AND conditioning (like Diffusers)
        final_config = Config(
            height=use_height,
            width=use_width,
            image_path=config.image_path,
            num_inference_steps=config.num_inference_steps,
            guidance=config.guidance,
        )
        runtime_config = RuntimeConfig(final_config, self.model_config)

        vl_width = use_width
        vl_height = use_height
        time_steps = tqdm(range(runtime_config.init_time_step, runtime_config.num_inference_steps))

        # 1. Create the initial latents
        latents = LatentCreator.create(
            seed=seed,
            height=runtime_config.height,
            width=runtime_config.width,
        )

        # 2. Generate prompt embeddings with comprehensive debugging (compare with diffusers_reference)
        prompt_embeds, prompt_mask, negative_prompt_embeds, negative_prompt_mask = self._debug_encode_prompts_with_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_path=runtime_config.image_path,
            runtime_config=runtime_config,
            vl_width=int(vl_width),
            vl_height=int(vl_height),
        )

        # 3. Generate image conditioning latents with MLX (compare with diffusers)
        static_image_latents, qwen_image_ids, cond_h_patches, cond_w_patches = QwenEditUtil.create_image_conditioning_latents(
            vae=self.vae,
            height=runtime_config.height,
            width=runtime_config.width,
            image_path=runtime_config.image_path,
            vl_width=int(vl_width),
            vl_height=int(vl_height),
        )

        # (Optional) Call subscribers for beginning of loop
        Callbacks.before_loop(
            seed=seed,
            prompt=prompt,
            latents=latents,
            config=runtime_config,
        )

        for t in time_steps:
            try:
                # 4.t Concatenate the updated latents with the static image latents
                hidden_states = mx.concatenate([latents, static_image_latents], axis=1)
                hidden_states_neg = mx.concatenate([latents, static_image_latents], axis=1)

                # 5.t Predict the noise
                noise = self.transformer(
                    t=t,
                    config=runtime_config,
                    hidden_states=hidden_states,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_mask,
                    qwen_image_ids=qwen_image_ids,  # Pass image IDs for conditioning
                    cond_image_grid=(1, cond_h_patches, cond_w_patches),  # Use VAE conditioning grid, not VL grid
                )[:, : latents.shape[1]]
                noise_negative = self.transformer(
                    t=t,
                    config=runtime_config,
                    hidden_states=hidden_states_neg,
                    encoder_hidden_states=negative_prompt_embeds,
                    encoder_hidden_states_mask=negative_prompt_mask,
                    qwen_image_ids=qwen_image_ids,
                    cond_image_grid=(1, cond_h_patches, cond_w_patches),  # Use VAE conditioning grid, not VL grid
                )[:, : latents.shape[1]]
                guided_noise = QwenImage.compute_guided_noise(noise, noise_negative, runtime_config.guidance)

                # 6.t Take one denoise step
                dt = runtime_config.sigmas[t + 1] - runtime_config.sigmas[t]
                latents = latents + guided_noise * dt

                # (Optional) Call subscribers in-loop
                Callbacks.in_loop(
                    t=t,
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
                    t=t,
                    seed=seed,
                    prompt=prompt,
                    latents=latents,
                    config=runtime_config,
                    time_steps=time_steps,
                )
                raise StopImageGenerationException(f"Stopping image generation at step {t + 1}/{len(time_steps)}")

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

        # 2. Format prompts (same template as diffusers)
        template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
        drop_idx = 64  # Same as diffusers

        formatted_prompt = template.format(prompt)
        formatted_negative_prompt = template.format(negative_prompt)

        # 3. Tokenize positive prompt (pass raw prompt - tokenizer will apply template)
        pos_input_ids, pos_attention_mask, pos_pixel_values, pos_image_grid_thw = tokenizer.tokenize_with_image(
            prompt, image, vl_width=vl_width, vl_height=vl_height
        )

        # 4. Run text encoder on positive prompt
        pos_hidden_states = self.qwen_vl_encoder(
            input_ids=pos_input_ids,
            attention_mask=pos_attention_mask,
            pixel_values=pos_pixel_values,
            image_grid_thw=pos_image_grid_thw,
        )

        # 5. Tokenize and encode negative prompt (pass raw prompt - tokenizer will apply template)
        print("🔧 MLX DEBUG: Tokenizing and encoding negative prompt...")
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
