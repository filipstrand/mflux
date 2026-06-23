"""Krea-2 text-to-image pipeline (text-only).

Denoising loop (er_sde by default, euler fallback) over the Krea-2 single-stream
DiT, conditioned on the Qwen3-VL-4B 12-layer tap (prefix-stripped), decoded with
the Qwen-Image VAE.

TODO(krea2): vision/edit path (see NOTES.md).
"""

import time
from pathlib import Path

import mlx.core as mx
from mlx import nn

from mflux.models.common.config import ModelConfig
from mflux.models.common.config.config import Config
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.krea2.krea2_initializer import Krea2Initializer
from mflux.models.krea2.model.krea2_sampler import flow_sigmas, make_stepper
from mflux.models.krea2.model.krea2_text_encoder.text_encoder import Krea2TextEncoder
from mflux.models.krea2.model.krea2_transformer.transformer import Krea2Transformer
from mflux.models.krea2.weights.krea2_weight_definition import Krea2WeightDefinition
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil


class Krea2(nn.Module):
    vae: QwenVAE
    transformer: Krea2Transformer
    text_encoder: Krea2TextEncoder

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        model_config: ModelConfig = None,  # noqa: RUF013 - resolved to ModelConfig.krea2() below
    ):
        super().__init__()
        Krea2Initializer.init(
            model=self,
            model_config=model_config or ModelConfig.krea2(),
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
        guidance: float = 1.0,
        negative_prompt: str | None = None,
        shift: float = 1.15,
        image_path: Path | str | None = None,
        image_strength: float | None = None,
        scheduler: str | None = None,
    ) -> GeneratedImage:
        # Reference turbo sampler is er_sde; "euler" is the plain flow fallback.
        sampler_name = scheduler if scheduler in ("er_sde", "euler") else "er_sde"
        if image_path is not None:
            raise NotImplementedError("Krea-2 image conditioning (edit/reference) is not wired yet — see NOTES.md.")

        config = Config(
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance=guidance,
        )
        # VAE is 8x spatial; DiT patch is 2. Use the 16-aligned dims from Config.
        lh, lw = config.height // 8, config.width // 8

        t0 = time.time()

        # 1. Initial noise latent (B, 16, h, w)
        latents = mx.random.normal((1, 16, lh, lw), key=mx.random.key(seed))

        # 2. Encode prompt (12-layer tap -> (B, seq, 12*2560))
        tok = self.tokenizers["qwen3vl"]
        pos = tok.tokenize(prompt)
        embeds = self.text_encoder.get_prompt_embeds(pos.input_ids, pos.attention_mask)
        do_cfg = guidance != 1.0
        if do_cfg:
            neg = tok.tokenize(negative_prompt or " ")
            neg_embeds = self.text_encoder.get_prompt_embeds(neg.input_ids, neg.attention_mask)

        # 3. Denoising loop (timestep = sigma in [0, 1]). Iterate config.time_steps (tqdm)
        #    so the standard progress bar shows; per-step mx.eval advances it live.
        sigmas = flow_sigmas(num_inference_steps, shift)
        stepper = make_stepper(sampler_name, sigmas, seed)
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)
        for t in config.time_steps:
            try:
                ts = sigmas[t].reshape(1)
                v = self.transformer(latents, ts, embeds)
                if do_cfg:
                    v_neg = self.transformer(latents, ts, neg_embeds)
                    v = v_neg + guidance * (v - v_neg)
                denoised = latents - sigmas[t] * v  # flow x0
                latents = stepper.step(t, latents, v, denoised)
                ctx.in_loop(t, latents)
                mx.eval(latents)
            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(f"Stopping image generation at step {t + 1}/{num_inference_steps}")
        ctx.after_loop(latents)

        # 4. Decode (Qwen-Image VAE denormalizes internally) and wrap as GeneratedImage
        decoded = self.vae.decode(latents)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            generation_time=time.time() - t0,
            negative_prompt=negative_prompt,
        )

    def save_model(self, base_path: str) -> None:
        """Save the (quantized) weights to disk for fast reload without re-quantizing."""
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=Krea2WeightDefinition,
        )
