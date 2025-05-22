import mlx.core as mx
from mlx import nn
from tqdm import tqdm

from mflux.callbacks.callbacks import Callbacks
from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.config.runtime_config import RuntimeConfig
from mflux.error.exceptions import StopImageGenerationException
from mflux.flux.flux_initializer import FluxInitializer
from mflux.flux_tools.fill.mask_util import MaskUtil
from mflux.latent_creator.latent_creator import LatentCreator
from mflux.models.text_encoder.clip_encoder.clip_encoder import CLIPEncoder
from mflux.models.text_encoder.prompt_encoder import PromptEncoder
from mflux.models.text_encoder.t5_encoder.t5_encoder import T5Encoder
from mflux.models.transformer.transformer import Transformer
from mflux.models.vae.vae import VAE
from mflux.post_processing.array_util import ArrayUtil
from mflux.post_processing.generated_image import GeneratedImage
from mflux.post_processing.image_util import ImageUtil


class Flux1Fill(nn.Module):
    vae: VAE
    transformer: Transformer
    t5_text_encoder: T5Encoder
    clip_text_encoder: CLIPEncoder

    def __init__(
        self,
        quantize: int | None = None,
        local_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ):
        super().__init__()
        FluxInitializer.init(
            flux_model=self,
            model_config=ModelConfig.dev_fill(),
            quantize=quantize,
            local_path=local_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        config: Config,
        reference_garment_path: str | None = None,
    ) -> GeneratedImage:
        # 0. Create a new runtime config based on the model type and input parameters
        config = RuntimeConfig(config, self.model_config)

        # For virtual try-on with side-by-side approach, double the width
        original_width = config.width
        if reference_garment_path:
            config.width = original_width * 2

        time_steps = tqdm(range(config.init_time_step, config.num_inference_steps))

        # 1. Create the initial latents and store the original noise for reference blending
        static_noise = LatentCreator.create(
            seed=seed,
            height=config.height,
            width=config.width,
        )
        latents = mx.array(static_noise)  # Create a copy

        # 2. Encode the prompt
        prompt_embeds, pooled_prompt_embeds = PromptEncoder.encode_prompt(
            prompt=prompt,
            prompt_cache=self.prompt_cache,
            t5_tokenizer=self.t5_tokenizer,
            clip_tokenizer=self.clip_tokenizer,
            t5_text_encoder=self.t5_text_encoder,
            clip_text_encoder=self.clip_text_encoder,
        )

        # 3. Prepare garment reference and masked latents
        # For virtual try-on, encode both the garment and create masked model latents
        garment_encoded, masked_model_latents, static_masked_latents = self.create_tryon_latents(
            config=config,
            original_width=original_width,
            img_path=config.image_path,
            mask_path=config.masked_image_path,
            garment_path=reference_garment_path,
        )

        # (Optional) Call subscribers for beginning of loop
        Callbacks.before_loop(
            seed=seed,
            prompt=prompt,
            latents=latents,
            config=config,
        )

        for t in time_steps:
            try:
                # 4.t Concatenate the updated latents with the static masked latents
                hidden_states = mx.concatenate([latents, static_masked_latents], axis=-1)

                # 5.t Predict the noise
                noise = self.transformer(
                    t=t,
                    config=config,
                    hidden_states=hidden_states,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                )

                # 6.t Take one denoise step
                dt = config.sigmas[t + 1] - config.sigmas[t]
                latents += noise * dt

                # 7.t If using garment reference, update the left side to maintain clear reference
                latents = self._update_garment_reference(
                    t=t,
                    config=config,
                    latents=latents,
                    garment_encoded=garment_encoded,
                    static_noise=static_noise,
                    original_width=original_width,
                )

                # (Optional) Call subscribers in-loop
                Callbacks.in_loop(
                    t=t,
                    seed=seed,
                    prompt=prompt,
                    latents=latents,
                    config=config,
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
                    config=config,
                    time_steps=time_steps,
                )
                raise StopImageGenerationException(f"Stopping image generation at step {t + 1}/{len(time_steps)}")

        # (Optional) Call subscribers after loop
        Callbacks.after_loop(
            seed=seed,
            prompt=prompt,
            latents=latents,
            config=config,
        )

        # 8. Decode the latent array and return the image
        latents = ArrayUtil.unpack_latents(latents=latents, height=config.height, width=config.width)
        decoded = self.vae.decode(latents)
        result = ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            image_path=config.image_path,
            image_strength=config.image_strength,
            masked_image_path=config.masked_image_path,
            reference_garment_path=reference_garment_path,
            generation_time=time_steps.format_dict["elapsed"],
        )

        # If this was a virtual try-on, return only the right half (the try-on result)
        if reference_garment_path:
            return result.get_right_half()
        return result

    def create_tryon_latents(self, config, original_width, img_path, mask_path, garment_path):
        """Create spatially concatenated latents for virtual try-on."""
        # 1. Load and process the model image
        model_image = ImageUtil.scale_to_dimensions(
            image=ImageUtil.load_image(img_path).convert("RGB"),
            target_width=original_width,
            target_height=config.height,
        )
        model_array = ImageUtil.to_array(model_image)

        # 2. Load and process the garment image
        garment_image = ImageUtil.scale_to_dimensions(
            image=ImageUtil.load_image(garment_path).convert("RGB"),
            target_width=original_width,
            target_height=config.height,
        )
        garment_array = ImageUtil.to_array(garment_image)

        # 3. Load and process the mask (for the model image)
        mask_image = ImageUtil.scale_to_dimensions(
            image=ImageUtil.load_image(mask_path).convert("RGB"),
            target_width=original_width,
            target_height=config.height,
        )
        mask_array = ImageUtil.to_array(mask_image, is_mask=True)

        # 4. Create empty mask for garment side
        garment_mask = mx.zeros_like(mask_array)

        # 5. Create concatenated image: [garment | masked_model]
        masked_model_array = model_array * (1 - mask_array)  # Apply mask to model image

        # Concatenate along width dimension (dim=3)
        concatenated_image = mx.concatenate([garment_array, masked_model_array], axis=3)
        concatenated_mask = mx.concatenate([garment_mask, mask_array], axis=3)

        # 6. Encode the garment separately for reference updates
        garment_encoded = self.vae.encode(garment_array)

        # 7. Encode the masked model separately (for reference)
        masked_model_encoded = self.vae.encode(masked_model_array)

        # 8. Encode the concatenated image
        encoded_image = self.vae.encode(concatenated_image)
        encoded_image = ArrayUtil.pack_latents(latents=encoded_image, height=config.height, width=config.width)

        # 9. Process the concatenated mask
        mask = MaskUtil._reshape_mask(the_mask=concatenated_mask, height=config.height, width=config.width)
        mask = ArrayUtil.pack_latents(latents=mask, height=config.height, width=config.width, num_channels_latents=64)

        # 10. Concatenate the encoded image and mask
        static_masked_latents = mx.concatenate([encoded_image, mask], axis=-1)

        # Return the garment encoding (for reference updates) and the static masked latents
        return garment_encoded, masked_model_encoded, static_masked_latents

    def _update_garment_reference(
        self,
        t: int,
        config: RuntimeConfig,
        latents: mx.array,
        garment_encoded: mx.array,
        static_noise: mx.array,
        original_width: int,
    ) -> mx.array:
        """Update the left side of the latents with a controlled blend of garment and noise."""
        # 1. Unpack the latents
        unpacked = ArrayUtil.unpack_latents(latents=latents, height=config.height, width=config.width)
        unpacked_static_noise = ArrayUtil.unpack_latents(latents=static_noise, height=config.height, width=config.width)

        # 2. Calculate latent width for garment (left half)
        latent_width = original_width // 8

        # 3. Override the left side with the reference garment blended with appropriate noise for current timestep
        # This uses the exact same approach as in-context LoRA to maintain reference clarity
        unpacked[:, :, :, 0:latent_width] = LatentCreator.add_noise_by_interpolation(
            clean=garment_encoded,
            noise=unpacked_static_noise[:, :, :, 0:latent_width],
            sigma=config.sigmas[t + 1],
        )

        # 4. Repack the latents
        return ArrayUtil.pack_latents(latents=unpacked, height=config.height, width=config.width)
