from pathlib import Path

import mlx.core as mx
from mlx import nn

from mflux.models.common.config import ModelConfig
from mflux.models.common.config.config import Config
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.depth_pro.model.depth_pro import DepthPro
from mflux.models.krea2.krea2_initializer import Krea2Initializer
from mflux.models.krea2.latent_creator.krea2_latent_creator import Krea2LatentCreator
from mflux.models.krea2.model.krea2_sampler import Krea2Sampler
from mflux.models.krea2.model.krea2_text_encoder.prompt_encoder import Krea2PromptEncoder
from mflux.models.krea2.model.krea2_text_encoder.text_encoder import Krea2TextEncoder
from mflux.models.krea2.model.krea2_transformer.transformer import Krea2Transformer
from mflux.models.krea2.variants.controlnet.krea2_depth_util import Krea2DepthUtil
from mflux.models.krea2.variants.txt2img.krea2 import Krea2
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
from mflux.utils.apple_silicon import AppleSiliconUtil
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil


class Krea2Depth(nn.Module):
    # Krea 2 with the depth ControlNet merged in: a wider input projection plus attention/MLP deltas,
    # conditioned on a depth latent that joins the image latent at every step.

    vae: QwenVAE
    transformer: Krea2Transformer
    text_encoder: Krea2TextEncoder

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        model_config: ModelConfig | None = None,
        controlnet_path: str | None = None,
        controlnet_strength: float = 1.0,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        bake_lora: bool = True,
    ):
        super().__init__()
        Krea2Initializer.init_depth(
            model=self,
            model_config=model_config or ModelConfig.krea2_depth_controlnet(),
            quantize=quantize,
            controlnet_path=controlnet_path,
            controlnet_strength=controlnet_strength,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            bake_lora=bake_lora,
        )
        # Loaded on first use, so a run with a ready-made depth map never loads Depth Pro.
        self.depth_pro: DepthPro | None = None

    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int = 8,
        height: int = 1024,
        width: int = 1024,
        guidance: float = 1.0,
        negative_prompt: str | None = None,
        image_path: Path | str | None = None,
        depth_image_path: Path | str | None = None,
        scheduler: str | None = None,
    ) -> GeneratedImage:
        resolved_scheduler = Krea2._resolve_scheduler(scheduler)
        config = Config(
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance=guidance,
            image_path=image_path,
            scheduler=resolved_scheduler,
        )

        sigmas = config.scheduler.sigmas
        latents = Krea2LatentCreator.create_noise(seed, config.height, config.width)
        control, depth_image = Krea2DepthUtil.encode_depth_control(
            vae=self.vae,
            depth_pro=None if depth_image_path else self._get_depth_pro(),
            width=config.width,
            height=config.height,
            image_path=image_path,
            depth_image_path=depth_image_path,
            tiling_config=self.tiling_config,
        )
        embeds, neg_embeds = Krea2PromptEncoder.encode_prompt_pair(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance=guidance,
            tokenizer=self.tokenizers["qwen3vl"],
            text_encoder=self.text_encoder,
            prompt_cache=self.prompt_cache,
        )
        mx.eval(latents, control, embeds)
        if neg_embeds is not None:
            mx.eval(neg_embeds)

        stepper = Krea2Sampler.make_stepper(resolved_scheduler, sigmas, seed)
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents, depth_image=depth_image)
        predict = Krea2Depth._predict(self.transformer, embeds, neg_embeds, guidance, control)

        for t in config.time_steps:
            try:
                ts = sigmas[t].reshape(1)
                v = predict(latents=latents, timestep=ts)
                denoised = latents - sigmas[t] * v
                latents = stepper.step(t, latents, v, denoised)
                ctx.in_loop(t, latents, denoised=denoised)
                mx.eval(latents)
            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )
        # The predict closure holds the transformer. Delete it so that --low-ram can free the transformer before decode.
        del predict
        ctx.after_loop(latents)

        decoded = VAEUtil.decode(vae=self.vae, latent=latents, tiling_config=self.tiling_config)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            generation_time=config.time_steps.format_dict["elapsed"],
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            negative_prompt=negative_prompt,
            image_path=config.image_path,
            depth_image_path=depth_image_path,
        )

    def _get_depth_pro(self) -> DepthPro:
        if self.depth_pro is None:
            self.depth_pro = DepthPro()
        return self.depth_pro

    @staticmethod
    def _predict(
        transformer: Krea2Transformer,
        embeds: mx.array,
        neg_embeds: mx.array | None,
        guidance: float,
        control: mx.array,
    ):
        def predict(latents: mx.array, timestep: mx.array) -> mx.array:
            v = transformer(latents, timestep, embeds, control=control)
            if neg_embeds is not None:
                v_neg = transformer(latents, timestep, neg_embeds, control=control)
                v = v_neg + guidance * (v - v_neg)
            return v

        if AppleSiliconUtil.is_m1_or_m2():
            return predict
        return mx.compile(predict)
