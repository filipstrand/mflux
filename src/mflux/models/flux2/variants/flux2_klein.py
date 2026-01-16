import time
from pathlib import Path

import mlx.core as mx
from mlx import nn

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.schedulers.flow_match_euler_discrete_scheduler import FlowMatchEulerDiscreteScheduler
from mflux.models.flux2.flux2_initializer import Flux2Initializer
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
from mflux.models.flux2.model.flux2_text_encoder.prompt_encoder import Flux2PromptEncoder
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil


class Flux2Klein(nn.Module):
    vae: Flux2VAE
    transformer: Flux2Transformer
    text_encoder: Qwen3TextEncoder

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        model_config: ModelConfig | None = None,
    ):
        super().__init__()
        Flux2Initializer.init(
            model=self,
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            model_config=model_config or ModelConfig.flux2_klein_4b(),
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int = 4,
        height: int = 1024,
        width: int = 1024,
        guidance: float = 1.0,
        image_path: Path | str | None = None,
        image_strength: float | None = None,
        scheduler: str = "flow_match_euler_discrete",
        negative_prompt: str | None = None,
    ) -> GeneratedImage:
        config = Config(
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance=guidance,
            image_path=image_path,
            image_strength=image_strength,
            scheduler=scheduler,
        )
        start_time = time.time()

        # 1. Encode prompt
        prompt_embeds, text_ids = Flux2PromptEncoder.encode_prompt(
            prompt=prompt,
            tokenizer=self.tokenizers["qwen3"],
            text_encoder=self.text_encoder,
            num_images_per_prompt=1,
            max_sequence_length=512,
            text_encoder_out_layers=(9, 18, 27),
        )

        negative_prompt_embeds = None
        negative_text_ids = None
        if config.guidance > 1.0 and negative_prompt:
            negative_prompt_embeds, negative_text_ids = Flux2PromptEncoder.encode_prompt(
                prompt=negative_prompt,
                tokenizer=self.tokenizers["qwen3"],
                text_encoder=self.text_encoder,
                num_images_per_prompt=1,
                max_sequence_length=512,
                text_encoder_out_layers=(9, 18, 27),
            )

        # 2. Prepare latents
        latents, latent_ids, latent_height, latent_width = Flux2LatentCreator.prepare_latents(
            seed=seed,
            height=config.height,
            width=config.width,
            batch_size=1,
        )
        latents = Flux2LatentCreator.pack_latents(latents)

        # 3. Prepare timesteps and sigmas
        image_seq_len = latents.shape[1]
        timesteps, sigmas = FlowMatchEulerDiscreteScheduler.get_timesteps_and_sigmas(
            image_seq_len=image_seq_len,
            num_inference_steps=config.num_inference_steps,
        )

        # 4. Denoising loop
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)
        for i in config.time_steps:
            try:
                t = timesteps[i]
                timestep = mx.full((latents.shape[0],), t, dtype=mx.float32)
                noise_pred = self.transformer(
                    hidden_states=latents,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep / 1000,
                    img_ids=latent_ids,
                    txt_ids=text_ids,
                    guidance=None,
                )

                if config.guidance > 1.0 and negative_prompt_embeds is not None and negative_text_ids is not None:
                    neg_noise_pred = self.transformer(
                        hidden_states=latents,
                        encoder_hidden_states=negative_prompt_embeds,
                        timestep=timestep / 1000,
                        img_ids=latent_ids,
                        txt_ids=negative_text_ids,
                        guidance=None,
                    )
                    noise_pred = neg_noise_pred + config.guidance * (noise_pred - neg_noise_pred)

                dt = sigmas[i + 1] - sigmas[i]
                latents = latents + dt.astype(latents.dtype) * noise_pred.astype(latents.dtype)

                ctx.in_loop(i, latents)
                mx.eval(latents)
            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(i, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {i + 1}/{config.num_inference_steps}"
                )

        ctx.after_loop(latents)

        # 5. Decode latents
        height_tokens = latent_height
        width_tokens = latent_width

        packed_latents = latents.reshape(latents.shape[0], height_tokens, width_tokens, latents.shape[-1]).transpose(
            0, 3, 1, 2
        )
        decoded = self.vae.decode_packed_latents(packed_latents)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=getattr(self, "bits", 0) or 0,
            generation_time=time.time() - start_time,
            negative_prompt=negative_prompt,
        )
