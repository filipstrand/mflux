import mlx.core as mx
from mlx import nn
from tqdm import tqdm

from mflux.callbacks.callbacks import Callbacks
from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.config.runtime_config import RuntimeConfig
from mflux.error.exceptions import StopImageGenerationException
from mflux.latent_creator.latent_creator import LatentCreator
from mflux.models.qwen.model.qwen_text_encoder.qwen_prompt_encoder import QwenPromptEncoder
from mflux.models.qwen.model.qwen_text_encoder.qwen_text_encoder import QwenTextEncoder
from mflux.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
from mflux.post_processing.array_util import ArrayUtil
from mflux.post_processing.generated_image import GeneratedImage
from mflux.post_processing.image_util import ImageUtil
from mflux.models.qwen.qwen_initializer import QwenImageInitializer


class QwenImage(nn.Module):
    vae: QwenVAE
    transformer: QwenTransformer
    text_encoder: QwenTextEncoder

    def __init__(
        self,
        model_config: ModelConfig,
        quantize: int | None = None,
        local_path: str | None = None,
    ):
        super().__init__()
        QwenImageInitializer.init(
            qwen_model=self,
            model_config=model_config,
            quantize=quantize,
            local_path=local_path,
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        config: Config,
        negative_prompt: str | None = None,
        prompt_embeds: mx.array | None = None,
        prompt_mask: mx.array | None = None,
        negative_prompt_embeds: mx.array | None = None,
        negative_prompt_mask: mx.array | None = None,
    ) -> GeneratedImage:
        # 0. Create a new runtime config based on the model type and input parameters
        runtime_config = RuntimeConfig(config, self.model_config)
        time_steps = tqdm(range(runtime_config.init_time_step, runtime_config.num_inference_steps))

        # 1. Create the initial latents
        latents = LatentCreator.create(
            seed=seed,
            height=runtime_config.height,
            width=runtime_config.width,
        )

        # 2. Encode the prompt
        prompt_embeds, prompt_mask, negative_prompt_embeds, negative_prompt_mask = QwenPromptEncoder.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_cache=self.prompt_cache,
            qwen_tokenizer=self.qwen_tokenizer,
            qwen_text_encoder=self.text_encoder,
        )
        del self.text_encoder

        # (Optional) Call subscribers for beginning of loop
        Callbacks.before_loop(
            seed=seed,
            prompt=prompt,
            latents=latents,
            config=runtime_config,
        )

        for t in time_steps:
            try:
                # 3. Predict the noise
                noise = self.transformer(
                    t=t,
                    config=runtime_config,
                    hidden_states=latents,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_mask,
                )
                noise_negative = self.transformer(
                    t=t,
                    config=runtime_config,
                    hidden_states=latents,
                    encoder_hidden_states=negative_prompt_embeds,
                    encoder_hidden_states_mask=negative_prompt_mask,
                )
                guided_noise = QwenImage._compute_guided_noise(noise, noise_negative, runtime_config.guidance)

                # 4.t Take one denoise step
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
        decoded = self.vae.decode_latents(latents)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=runtime_config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            lora_paths=[],
            lora_scales=[],
            image_path=runtime_config.image_path,
            image_strength=runtime_config.image_strength,
            generation_time=time_steps.format_dict["elapsed"],
        )

    @staticmethod
    def _compute_guided_noise(
        noise: mx.array,
        noise_negative: mx.array,
        guidance: float,
    ) -> mx.array:
        combined = noise_negative + guidance * (noise - noise_negative)
        cond_norm = mx.sqrt(mx.sum(noise * noise, axis=-1, keepdims=True) + 1e-12)
        noise_norm = mx.sqrt(mx.sum(combined * combined, axis=-1, keepdims=True) + 1e-12)
        noise = combined * (cond_norm / noise_norm)
        return noise
