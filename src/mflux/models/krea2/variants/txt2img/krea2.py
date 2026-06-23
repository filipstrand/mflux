from pathlib import Path

import mlx.core as mx
from mlx import nn
from PIL import Image

from mflux.models.common.config import ModelConfig
from mflux.models.common.config.config import Config
from mflux.models.common.latent_creator.latent_creator import Img2Img, LatentCreator
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.krea2.krea2_initializer import Krea2Initializer
from mflux.models.krea2.latent_creator import Krea2LatentCreator
from mflux.models.krea2.model.krea2_text_encoder.prompt_encoder import Krea2PromptEncoder
from mflux.models.krea2.model.krea2_text_encoder.text_encoder import Krea2TextEncoder
from mflux.models.krea2.model.krea2_transformer.transformer import Krea2Transformer
from mflux.models.krea2.weights.krea2_weight_definition import Krea2WeightDefinition
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
from mflux.utils.apple_silicon import AppleSiliconUtil
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.image_util import ImageUtil


class Krea2Image(nn.Module):
    vae: QwenVAE
    text_encoder: Krea2TextEncoder
    transformer: Krea2Transformer

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        model_config: ModelConfig = ModelConfig.krea2_turbo(),
    ):
        super().__init__()
        Krea2Initializer.init(
            model=self,
            model_config=model_config,
            quantize=quantize,
            model_path=model_path,
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int = 8,
        height: int = 1024,
        width: int = 1024,
        guidance: float | None = None,
        image_path: Path | str | None = None,
        image_strength: float | None = None,
        scheduler: str | None = None,
        negative_prompt: str | None = None,
    ) -> Image.Image:
        if scheduler is None:
            scheduler = "flow_match_euler_discrete"
        if not self.model_config.supports_guidance:
            if guidance is not None and guidance != 0.0:
                raise ValueError("Krea-2 Turbo does not support classifier-free guidance. Use guidance=0.0.")
            guidance = 0.0
        elif guidance is None:
            guidance = 3.5

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
        self._configure_scheduler(config)
        latents = LatentCreator.create_for_txt2img_or_img2img(
            seed=seed,
            width=config.width,
            height=config.height,
            img2img=Img2Img(
                vae=self.vae,
                latent_creator=Krea2LatentCreator,
                sigmas=config.scheduler.sigmas,
                init_time_step=config.init_time_step,
                image_path=config.image_path,
                tiling_config=self.tiling_config,
            ),
        )
        text_encodings, text_mask, negative_encodings, negative_mask = self._encode_prompts(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance=config.guidance,
        )
        position_ids = Krea2Transformer.position_ids(
            text_seq_len=text_encodings.shape[1],
            image_seq_len=latents.shape[1],
            height=config.height,
            width=config.width,
        )
        mx.eval(latents, text_encodings, text_mask, position_ids)

        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)
        predict = self._predict(self.transformer)

        for t in config.time_steps:
            try:
                timestep = config.scheduler.sigmas[t].reshape((1,))
                noise = predict(
                    latents=latents,
                    timestep=timestep,
                    position_ids=position_ids,
                    text_encodings=text_encodings,
                    text_mask=text_mask,
                    negative_encodings=negative_encodings,
                    negative_mask=negative_mask,
                    guidance=config.guidance,
                )
                latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)
                ctx.in_loop(t, latents)
                mx.eval(latents)

            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )

        ctx.after_loop(latents)
        decoded = self._decode_latents(latents=latents, config=config)
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

    def _configure_scheduler(self, config: Config) -> None:
        scheduler = config.scheduler
        if hasattr(scheduler, "set_mu"):
            scheduler.set_mu(1.15)

    def _encode_prompts(
        self,
        *,
        prompt: str,
        negative_prompt: str | None,
        guidance: float,
    ) -> tuple[mx.array, mx.array, mx.array | None, mx.array | None]:
        text_encodings, text_mask = Krea2PromptEncoder.encode_prompt(
            prompt=prompt,
            prompt_cache=self.prompt_cache,
            tokenizer=self.tokenizers["krea2"],
            text_encoder=self.text_encoder,
            max_length=self.model_config.max_sequence_length or 512,
        )
        if guidance <= 0.0:
            return text_encodings, text_mask, None, None

        negative_text = negative_prompt if negative_prompt and negative_prompt.strip() else ""
        negative_encodings, negative_mask = Krea2PromptEncoder.encode_prompt(
            prompt=negative_text,
            prompt_cache=self.prompt_cache,
            tokenizer=self.tokenizers["krea2"],
            text_encoder=self.text_encoder,
            max_length=self.model_config.max_sequence_length or 512,
        )
        return text_encodings, text_mask, negative_encodings, negative_mask

    def _decode_latents(self, *, latents: mx.array, config: Config) -> mx.array:
        unpacked = Krea2LatentCreator.unpack_latents(latents=latents, height=config.height, width=config.width)
        return VAEUtil.decode(vae=self.vae, latent=unpacked, tiling_config=self.tiling_config)


    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=Krea2WeightDefinition.resolve(self.model_config),
        )

    @staticmethod
    def _predict(transformer: Krea2Transformer):
        def predict(
            latents: mx.array,
            timestep: mx.array,
            position_ids: mx.array,
            text_encodings: mx.array,
            text_mask: mx.array,
            negative_encodings: mx.array | None,
            negative_mask: mx.array | None,
            guidance: float,
        ) -> mx.array:
            timestep = mx.broadcast_to(timestep, (latents.shape[0],))
            noise = transformer(
                hidden_states=latents,
                encoder_hidden_states=text_encodings,
                timestep=timestep,
                position_ids=position_ids,
                encoder_attention_mask=text_mask,
            )
            if guidance <= 0.0 or negative_encodings is None or negative_mask is None:
                return noise

            negative_noise = transformer(
                hidden_states=latents,
                encoder_hidden_states=negative_encodings,
                timestep=timestep,
                position_ids=position_ids,
                encoder_attention_mask=negative_mask,
            )
            return noise + guidance * (noise - negative_noise)

        if AppleSiliconUtil.is_m1_or_m2():
            return predict
        return mx.compile(predict)
