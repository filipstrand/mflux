import math

import mlx.core as mx
from mlx import nn
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


class QwenImageEditPlus(nn.Module):
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
            model_config=ModelConfig.qwen_image_edit_plus(),
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
        image_paths: list[str] | None = None,
    ) -> GeneratedImage:
        if image_paths is None:
            image_paths = [config.image_path]

        runtime_config, vl_width, vl_height, vae_width, vae_height = self._compute_dimensions(config, image_paths)
        timesteps = runtime_config.scheduler.timesteps
        time_steps = tqdm(range(len(timesteps)))

        # 1. Create initial latents
        latents = LatentCreator.create(
            seed=seed,
            height=runtime_config.height,
            width=runtime_config.width,
        )

        # 2. Encode the prompt
        prompt_embeds, prompt_mask, negative_prompt_embeds, negative_prompt_mask = self._encode_prompts_with_image_plus(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_paths=image_paths,
            runtime_config=runtime_config,
            vl_width=vl_width,
            vl_height=vl_height,
        )

        # 3. Generate image conditioning latents
        static_image_latents, qwen_image_ids, cond_h_patches, cond_w_patches, num_images = (
            QwenEditUtil.create_image_conditioning_latents(
                vae=self.vae,
                height=vae_height,
                width=vae_width,
                image_paths=image_paths,
                vl_width=vl_width,
                vl_height=vl_height,
            )
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
                if num_images > 1:
                    cond_image_grid = [(1, cond_h_patches, cond_w_patches) for _ in range(num_images)]
                else:
                    cond_image_grid = (1, cond_h_patches, cond_w_patches)

                noise = self.transformer(
                    t=t,
                    config=runtime_config,
                    hidden_states=hidden_states,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_mask,
                    qwen_image_ids=qwen_image_ids,
                    cond_image_grid=cond_image_grid,
                )[:, : latents.shape[1]]
                noise_negative = self.transformer(
                    t=t,
                    config=runtime_config,
                    hidden_states=hidden_states_neg,
                    encoder_hidden_states=negative_prompt_embeds,
                    encoder_hidden_states_mask=negative_prompt_mask,
                    qwen_image_ids=qwen_image_ids,
                    cond_image_grid=cond_image_grid,
                )[:, : latents.shape[1]]
                guided_noise = QwenImage.compute_guided_noise(noise, noise_negative, runtime_config.guidance)

                # 6.t Take one denoise step
                latents = runtime_config.scheduler.step(
                    model_output=guided_noise,
                    timestep=t,
                    sample=latents,
                )

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
                raise StopImageGenerationException(f"Stopping image generation at step {t + 1}/{len(timesteps)}")

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

    def _encode_prompts_with_image_plus(
        self,
        prompt: str,
        negative_prompt: str,
        image_paths: list[str],
        runtime_config,
        vl_width: int | None = None,
        vl_height: int | None = None,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        tokenizer = self.qwen_vl_tokenizer
        pos_input_ids, pos_attention_mask, pos_pixel_values, pos_image_grid_thw = tokenizer.tokenize_with_image(
            prompt, image_paths, vl_width=vl_width, vl_height=vl_height
        )

        pos_hidden_states = self.qwen_vl_encoder(
            input_ids=pos_input_ids,
            attention_mask=pos_attention_mask,
            pixel_values=pos_pixel_values,
            image_grid_thw=pos_image_grid_thw,
        )
        mx.eval(pos_hidden_states[0])
        mx.eval(pos_hidden_states[1])

        neg_prompt = negative_prompt if negative_prompt is not None else ""

        # Use real MLX tokenizer for negative prompt
        neg_input_ids, neg_attention_mask, neg_pixel_values, neg_image_grid_thw = tokenizer.tokenize_with_image(
            neg_prompt, image_paths, vl_width=vl_width, vl_height=vl_height
        )

        neg_hidden_states = self.qwen_vl_encoder(
            input_ids=neg_input_ids,
            attention_mask=neg_attention_mask,
            pixel_values=neg_pixel_values,
            image_grid_thw=neg_image_grid_thw,
        )
        mx.eval(neg_hidden_states[0])
        mx.eval(neg_hidden_states[1])

        final_prompt_embeds = pos_hidden_states[0].astype(mx.float16)
        final_prompt_mask = pos_hidden_states[1].astype(mx.float16)

        return (
            final_prompt_embeds,  # prompt_embeds
            final_prompt_mask,  # prompt_mask
            neg_hidden_states[0].astype(mx.float16),  # negative_prompt_embeds
            neg_hidden_states[1].astype(mx.float16),  # negative_prompt_mask
        )

    def _compute_dimensions(
        self,
        config: Config,
        image_paths: list[str],
    ) -> tuple[RuntimeConfig, int, int, int, int]:
        last_image = ImageUtil.load_image(image_paths[-1]).convert("RGB")
        image_size = last_image.size

        target_area = 1024 * 1024
        ratio = image_size[0] / image_size[1]
        calculated_width = math.sqrt(target_area * ratio)
        calculated_height = calculated_width / ratio
        calculated_width = round(calculated_width / 32) * 32
        calculated_height = round(calculated_height / 32) * 32

        use_height = config.height or calculated_height
        use_width = config.width or calculated_width

        vae_scale_factor = 8
        multiple_of = vae_scale_factor * 2
        use_width = use_width // multiple_of * multiple_of
        use_height = use_height // multiple_of * multiple_of

        final_config = Config(
            height=use_height,
            width=use_width,
            image_path=config.image_path,
            num_inference_steps=config.num_inference_steps,
            guidance=config.guidance,
            scheduler=config.scheduler_str,
        )
        runtime_config = RuntimeConfig(final_config, self.model_config)

        CONDITION_IMAGE_SIZE = 384 * 384
        condition_ratio = image_size[0] / image_size[1]
        vl_width = math.sqrt(CONDITION_IMAGE_SIZE * condition_ratio)
        vl_height = vl_width / condition_ratio
        vl_width = round(vl_width / 32) * 32
        vl_height = round(vl_height / 32) * 32

        VAE_IMAGE_SIZE = 1024 * 1024
        vae_ratio = image_size[0] / image_size[1]
        vae_width = math.sqrt(VAE_IMAGE_SIZE * vae_ratio)
        vae_height = vae_width / vae_ratio
        vae_width = round(vae_width / 32) * 32
        vae_height = round(vae_height / 32) * 32

        return runtime_config, int(vl_width), int(vl_height), int(vae_width), int(vae_height)
