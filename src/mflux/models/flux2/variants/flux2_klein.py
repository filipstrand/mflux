import time
from pathlib import Path

import mlx.core as mx
from mlx import nn

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux2.flux2_initializer import Flux2Initializer
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
from mflux.models.flux2.model.flux2_text_encoder.prompt_encoder import Flux2PromptEncoder
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.models.flux2.schedulers.flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
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
            model_config=model_config or ModelConfig.from_name("flux2-klein-4b"),
        )

    def generate_image(self, **kwargs):
        seed: int = kwargs.get("seed", 0)
        prompt: str | list[str] = kwargs.get("prompt", "")
        num_inference_steps: int = kwargs.get("num_inference_steps", 4)
        height: int = kwargs.get("height", 1024)
        width: int = kwargs.get("width", 1024)
        guidance: float = kwargs.get("guidance", 1.0)
        scheduler: str = kwargs.get("scheduler", "flow_match_euler_discrete")
        negative_prompt: str | list[str] | None = kwargs.get("negative_prompt", "")
        latents_path: Path | str | None = kwargs.get("latents_path", None)
        start_time = time.time()

        # 1. Encode prompt
        prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            num_images_per_prompt=1,
            max_sequence_length=512,
            text_encoder_out_layers=(9, 18, 27),
        )

        negative_prompt_embeds = None
        negative_text_ids = None
        if guidance > 1.0 and negative_prompt is not None:
            negative_prompt_embeds, negative_text_ids = self.encode_prompt(
                prompt=negative_prompt,
                num_images_per_prompt=1,
                max_sequence_length=512,
                text_encoder_out_layers=(9, 18, 27),
            )

        # 2. Prepare latents
        if latents_path is not None:
            loaded = mx.load(str(latents_path))
            if isinstance(loaded, dict):
                latents = mx.array(next(iter(loaded.values())))
            else:
                latents = mx.array(loaded)
            latent_ids = Flux2LatentCreator.prepare_latent_ids_from_packed(latents)
        else:
            latents, latent_ids, latent_height, latent_width = Flux2LatentCreator.prepare_latents(
                seed=seed,
                height=height,
                width=width,
                batch_size=1,
            )
            latents = Flux2LatentCreator.pack_latents(latents)

        # 3. Prepare timesteps and sigmas
        image_seq_len = latents.shape[1]
        timesteps, sigmas = FlowMatchEulerDiscreteScheduler.get_timesteps_and_sigmas(
            image_seq_len=image_seq_len,
            num_inference_steps=num_inference_steps,
        )

        # 4. Denoising loop
        batch_size = latents.shape[0]
        for i in range(num_inference_steps):
            t = timesteps[i]
            timestep = mx.full((batch_size,), t, dtype=mx.float32)
            noise_pred = self.transformer(
                hidden_states=latents,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep / 1000,
                img_ids=latent_ids,
                txt_ids=text_ids,
                guidance=None,
            )

            if guidance > 1.0 and negative_prompt_embeds is not None and negative_text_ids is not None:
                neg_noise_pred = self.transformer(
                    hidden_states=latents,
                    encoder_hidden_states=negative_prompt_embeds,
                    timestep=timestep / 1000,
                    img_ids=latent_ids,
                    txt_ids=negative_text_ids,
                    guidance=None,
                )
                noise_pred = neg_noise_pred + guidance * (noise_pred - neg_noise_pred)

            dt = sigmas[i + 1] - sigmas[i]
            latents = latents + dt.astype(latents.dtype) * noise_pred.astype(latents.dtype)

        # 5. Decode latents
        if latents_path is None:
            height_tokens = latent_height
            width_tokens = latent_width
        else:
            height_tokens = int(mx.max(latent_ids[:, :, 1]).item()) + 1
            width_tokens = int(mx.max(latent_ids[:, :, 2]).item()) + 1

        packed_latents = latents.reshape(batch_size, height_tokens, width_tokens, latents.shape[-1]).transpose(
            0, 3, 1, 2
        )
        decoded = self.vae.decode_packed_latents(packed_latents)
        config = Config(
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance=guidance,
            scheduler=scheduler,
        )
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt if isinstance(prompt, str) else "\n".join(prompt),
            quantization=getattr(self, "bits", 0) or 0,
            generation_time=time.time() - start_time,
            negative_prompt=negative_prompt if isinstance(negative_prompt, str) else None,
        )

    def encode_prompt(
        self,
        prompt: str | list[str],
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        text_encoder_out_layers: tuple[int, ...] = (9, 18, 27),
    ) -> tuple[mx.array, mx.array]:
        return Flux2PromptEncoder.encode_prompt(
            prompt=prompt,
            tokenizer=self.tokenizers["qwen3"],
            text_encoder=self.text_encoder,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )
