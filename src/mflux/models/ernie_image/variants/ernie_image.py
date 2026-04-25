from pathlib import Path

import mlx.core as mx
from mlx import nn
from PIL import Image

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.latent_creator.latent_creator import Img2Img, LatentCreator
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.ernie_image.ernie_image_initializer import ErnieImageInitializer
from mflux.models.ernie_image.latent_creator import ErnieLatentCreator
from mflux.models.ernie_image.model.ernie_text_encoder.prompt_encoder import ErniePromptEncoder
from mflux.models.ernie_image.model.ernie_text_encoder.text_encoder import ErnieMistralTextEncoder
from mflux.models.ernie_image.model.ernie_transformer.transformer import ErnieTransformer
from mflux.models.ernie_image.weights.ernie_weight_definition import ErnieWeightDefinition
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.utils.apple_silicon import AppleSiliconUtil
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.image_util import ImageUtil


class ErnieImage(nn.Module):
    vae: Flux2VAE
    text_encoder: ErnieMistralTextEncoder
    transformer: ErnieTransformer

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        model_config: ModelConfig = ModelConfig.ernie_image_turbo(),
    ):
        super().__init__()
        self._text_cache = None
        ErnieImageInitializer.init(
            model=self,
            model_config=model_config,
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int = 8,
        height: int = 1024,
        width: int = 1024,
        guidance: float = 1.0,
        image_path: Path | str | None = None,
        image_strength: float | None = None,
        scheduler: str | None = None,
        negative_prompt: str | None = None,
    ) -> Image.Image:
        if scheduler is None:
            scheduler = "linear"

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

        latents = LatentCreator.create_for_txt2img_or_img2img(
            seed=seed,
            width=config.width,
            height=config.height,
            img2img=Img2Img(
                vae=self.vae,
                latent_creator=ErnieLatentCreator,
                image_path=config.image_path,
                sigmas=config.scheduler.sigmas,
                init_time_step=config.init_time_step,
                tiling_config=self.tiling_config,
            ),
        )

        # Encode text — cache on (prompt, negative_prompt, guidance)
        cache_key = (prompt, negative_prompt, config.guidance)
        if self._text_cache is None or self._text_cache[0] != cache_key:
            text_bth, text_lens = self._encode_prompts(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance=config.guidance,
            )
            mx.eval(text_bth, text_lens)
            self._text_cache = (cache_key, text_bth, text_lens)
        else:
            _, text_bth, text_lens = self._text_cache

        mx.eval(latents)
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)
        predict = self._predict(self.transformer)

        for t in config.time_steps:
            try:
                sigma_t = config.scheduler.sigmas[t].reshape((1,))

                # Predict velocity (with CFG if guidance > 1)
                noise = self._predict_noise(
                    predict=predict,
                    latents=latents,
                    sigma=sigma_t,
                    text_bth=text_bth,
                    text_lens=text_lens,
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

        decoded = self.vae.decode_packed_latents(latents)
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
            model_path=self.model_path,
        )

    def _encode_prompts(
        self,
        *,
        prompt: str,
        negative_prompt: str | None,
        guidance: float,
    ) -> tuple[mx.array, mx.array]:
        """Returns (text_bth [B, T, 3072], text_lens [B]) for the denoising batch."""
        tokenizer = self.tokenizers["ernie"]
        if guidance <= 1.0:
            prompts = [prompt]
        else:
            neg = negative_prompt if negative_prompt and negative_prompt.strip() else " "
            prompts = [neg, prompt]

        return ErniePromptEncoder.build_text_batch(
            prompts=prompts,
            tokenizer=tokenizer,
            text_encoder=self.text_encoder,
        )

    @staticmethod
    def _predict(transformer: ErnieTransformer):
        cached = getattr(transformer, "_compiled_predict", None)
        if cached is not None:
            return cached

        def predict(latents: mx.array, timestep: mx.array, text_bth: mx.array, text_lens: mx.array) -> mx.array:
            return transformer(
                hidden_states=latents,
                timestep=timestep,
                text_bth=text_bth,
                text_lens=text_lens,
            )

        fn = mx.compile(predict) if AppleSiliconUtil.should_use_compile() else predict
        transformer._compiled_predict = fn
        return fn

    @staticmethod
    def _predict_noise(
        predict,
        latents: mx.array,
        sigma: mx.array,
        text_bth: mx.array,
        text_lens: mx.array,
        guidance: float,
    ) -> mx.array:
        timestep = sigma * 1000
        B = text_bth.shape[0]
        if B == 1:
            return predict(latents, mx.broadcast_to(timestep, (1,)), text_bth, text_lens)
        # CFG: text_bth is [2, T, H] with uncond first, cond second
        latent_input = mx.concatenate([latents, latents], axis=0)
        pred = predict(latent_input, mx.broadcast_to(timestep, (2,)), text_bth, text_lens)
        pred_uncond, pred_cond = pred[:1], pred[1:]
        return pred_uncond + guidance * (pred_cond - pred_uncond)

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=ErnieWeightDefinition,
        )
