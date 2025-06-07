import mlx.core as mx
from mlx import nn
from tqdm import tqdm

from mflux.callbacks.callbacks import Callbacks
from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.config.runtime_config import RuntimeConfig
from mflux.error.exceptions import StopImageGenerationException
from mflux.flux.flux_initializer import FluxInitializer
from mflux.latent_creator.latent_creator import LatentCreator
from mflux.models.text_encoder.clip_encoder.clip_encoder import CLIPEncoder
from mflux.models.text_encoder.prompt_encoder import PromptEncoder
from mflux.models.text_encoder.t5_encoder.t5_encoder import T5Encoder
from mflux.models.transformer.transformer import Transformer
from mflux.models.vae.vae import VAE
from mflux.post_processing.array_util import ArrayUtil
from mflux.post_processing.generated_image import GeneratedImage
from mflux.post_processing.image_util import ImageUtil


class Flux1InContextDev(nn.Module):
    vae: VAE
    transformer: Transformer
    t5_text_encoder: T5Encoder
    clip_text_encoder: CLIPEncoder

    def __init__(
        self,
        model_config: ModelConfig,
        quantize: int | None = None,
        local_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        lora_names: list[str] | None = None,
        lora_repo_id: str | None = None,
    ):
        super().__init__()
        FluxInitializer.init(
            flux_model=self,
            model_config=model_config,
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
    ) -> GeneratedImage:
        # 0. Create a new runtime config based on the model type and input parameters
        config = RuntimeConfig(config, self.model_config)
        time_steps = tqdm(range(config.init_time_step, config.num_inference_steps))

        # 1. Encode the reference image
        encoded_image = LatentCreator.encode_image(
            vae=self.vae,
            image_path=config.image_path,
            height=config.height,
            width=config.width,
        )

        # 2. Create the initial latents and keep the initial static noise for later blending
        static_noise = Flux1InContextDev._create_in_context_latents(seed=seed, config=config)
        latents = mx.array(static_noise)

        # 3. Encode the prompt
        prompt_embeds, pooled_prompt_embeds = PromptEncoder.encode_prompt(
            prompt=prompt,
            prompt_cache=self.prompt_cache,
            t5_tokenizer=self.t5_tokenizer,
            clip_tokenizer=self.clip_tokenizer,
            t5_text_encoder=self.t5_text_encoder,
            clip_text_encoder=self.clip_text_encoder,
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
                # 4.t Predict the noise
                noise = self.transformer(
                    t=t,
                    config=config,
                    hidden_states=latents,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                )

                # 5.t Take one denoise step and update latents
                dt = config.sigmas[t + 1] - config.sigmas[t]
                latents += noise * dt

                # 6.t Override the left-hand side of latents by linearly interpolating between latents and static noise
                latents = Flux1InContextDev._update_latents(
                    t=t,
                    config=config,
                    latents=latents,
                    encoded_image=encoded_image,
                    static_noise=static_noise,
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

        # 6. Decode the latent array and return the image
        latents = ArrayUtil.unpack_latents(latents=latents, height=config.height, width=config.width)
        decoded = self.vae.decode(latents)
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
            generation_time=time_steps.format_dict["elapsed"],
        )

    @staticmethod
    def _create_in_context_latents(seed: int, config: RuntimeConfig):
        # 1. Double the width for side-by-side generation
        config.width = 2 * config.width

        # 2. Create the initial latents with doubled width
        latent_height = config.height // 8
        latent_width = config.width // 8

        # 3. Create noise with appropriate dimensions
        static_noise = mx.random.normal(shape=[1, 16, latent_height, latent_width], key=mx.random.key(seed))
        latents = ArrayUtil.pack_latents(latents=static_noise, height=config.height, width=config.width)
        return latents

    @staticmethod
    def _update_latents(
        t: int,
        config: RuntimeConfig,
        latents: mx.array,
        encoded_image: mx.array,
        static_noise: mx.array,
    ) -> mx.array:
        # 1. Unpack the latents
        unpacked = ArrayUtil.unpack_latents(latents=latents, height=config.height, width=config.width)
        unpacked_static_noise = ArrayUtil.unpack_latents(latents=static_noise, height=config.height, width=config.width)

        # 2. Calculate latent_width from the config (original width is half of current width)
        latent_width = (config.width // 2) // 8

        # 3. Override the left side with the reference image blended with appropriate noise for current timestep
        unpacked[:, :, :, 0:latent_width] = LatentCreator.add_noise_by_interpolation(
            clean=encoded_image[:, :, :, 0:latent_width],
            noise=unpacked_static_noise[:, :, :, 0:latent_width],
            sigma=config.sigmas[t + 1],
        )

        # 4. Repack the latents
        return ArrayUtil.pack_latents(latents=unpacked, height=config.height, width=config.width)
