import math
import time

import mlx.core as mx
import numpy as np
from mlx import nn
from tqdm import tqdm

from mflux.callbacks.callbacks import Callbacks
from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.config.runtime_config import RuntimeConfig
from mflux.latent_creator.latent_creator import LatentCreator
from mflux.models.vae.qwen_vae import QwenImageVAE
from mflux.post_processing.array_util import ArrayUtil
from mflux.post_processing.generated_image import GeneratedImage
from mflux.post_processing.image_util import ImageUtil
from mflux.qwen.qwen_initializer import QwenImageInitializer
from mflux.qwen.qwen_prompt_encoder import QwenPromptEncoder


class QwenImage(nn.Module):
    vae: QwenImageVAE

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

    def decode_latents(self, latents: mx.array) -> mx.array:
        latents_mean = mx.array(
            [
                -0.7571,
                -0.7089,
                -0.9113,
                0.1075,
                -0.1745,
                0.9653,
                -0.1517,
                1.5508,
                0.4134,
                -0.0715,
                0.5517,
                -0.3632,
                -0.1922,
                -0.9497,
                0.2503,
                -0.2921,
            ]
        ).reshape(1, 16, 1, 1, 1)

        latents_std = mx.array(
            [
                2.8184,
                1.4541,
                2.3275,
                2.6558,
                1.2196,
                1.7708,
                2.6052,
                2.0743,
                3.2687,
                2.1526,
                2.8652,
                1.5579,
                1.6382,
                1.1253,
                2.8251,
                1.916,
            ]
        ).reshape(1, 16, 1, 1, 1)

        latents = latents / (1.0 / latents_std) + latents_mean

        return self.vae.decode(latents)

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
        reference_timesteps: mx.array | None = None,
    ) -> GeneratedImage:
        print("\n🚀 Starting image generation...")
        total_start = time.time()
        runtime_config = RuntimeConfig(config, self.model_config)
        time_steps = tqdm(range(runtime_config.init_time_step, runtime_config.num_inference_steps))

        # Encode prompts if not provided
        if prompt_embeds is None or prompt_mask is None:
            print("📝 Encoding text prompt...")
            encode_start = time.time()
            prompt_embeds, prompt_mask = QwenPromptEncoder.encode_prompt(
                prompt=prompt,
                prompt_cache=self.prompt_cache,
                qwen_tokenizer=self.qwen_tokenizer,
                qwen_text_encoder=self.text_encoder,
            )
            encode_time = time.time() - encode_start
            print(f"✅ Text encoding completed in {encode_time:.2f} seconds")

        if negative_prompt is not None and (negative_prompt_embeds is None or negative_prompt_mask is None):
            print("📝 Encoding negative prompt...")
            negative_prompt_embeds, negative_prompt_mask = QwenPromptEncoder.encode_negative_prompt(
                negative_prompt=negative_prompt,
                prompt_cache=self.prompt_cache,
                qwen_tokenizer=self.qwen_tokenizer,
                qwen_text_encoder=self.text_encoder,
            )

        # Almost necessary to do now to free up memory
        del self.text_encoder

        # 1. Create the initial latents
        latents = LatentCreator.create(
            seed=seed,
            height=runtime_config.height,
            width=runtime_config.width,
        )

        batch = latents.shape[0]
        num_patches = latents.shape[1]
        side = int(round(math.sqrt(num_patches)))
        if side * side != num_patches:
            raise ValueError(f"Latent patches are not square: {num_patches}")

        img_shapes = [(1, side, side)] * batch
        if prompt_mask is not None:
            txt_seq_lens = [int(mx.sum(prompt_mask[i]).item()) for i in range(prompt_mask.shape[0])]
        else:
            txt_seq_lens = [prompt_embeds.shape[1]] * batch

        # Handle reference timesteps if provided
        if reference_timesteps is not None:
            # Working version: sigmas = (t_ref / 1000.0).astype(np.float32)
            reference_sigmas = np.array((reference_timesteps / 1000.0).astype(mx.float32))
            reference_sigmas = np.append(reference_sigmas, 0.0)  # Add zero for dt calculation
            runtime_config.sigmas = mx.array(reference_sigmas).astype(mx.float32)

        Callbacks.before_loop(
            seed=seed,
            prompt=prompt,
            latents=latents,
            config=runtime_config,
        )

        print(f"🔄 Running {runtime_config.num_inference_steps} denoising steps...")
        denoising_start = time.time()
        for t in time_steps:
            try:
                # 3. Predict the noise
                noise = self.transformer(
                    t=t,
                    config=runtime_config,
                    hidden_states=latents,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    encoder_hidden_states_mask=prompt_mask,
                )

                if negative_prompt_embeds is not None:
                    neg_txt_seq_lens = txt_seq_lens
                    if negative_prompt_mask is not None:
                        neg_txt_seq_lens = [
                            int(mx.sum(negative_prompt_mask[i]).item()) for i in range(negative_prompt_mask.shape[0])
                        ]

                    neg_noise = self.transformer(
                        t=t,
                        config=runtime_config,
                        hidden_states=latents,
                        encoder_hidden_states=negative_prompt_embeds,
                        img_shapes=img_shapes,
                        txt_seq_lens=neg_txt_seq_lens,
                        encoder_hidden_states_mask=negative_prompt_mask,
                    )

                    cfg_scale = getattr(runtime_config, "guidance", 4.0)
                    combined = neg_noise + cfg_scale * (noise - neg_noise)

                    cond_norm = mx.sqrt(mx.sum(noise * noise, axis=-1, keepdims=True) + 1e-12)
                    noise_norm = mx.sqrt(mx.sum(combined * combined, axis=-1, keepdims=True) + 1e-12)
                    noise = combined * (cond_norm / noise_norm)

                # Take one denoise step
                dt = runtime_config.sigmas[t + 1] - runtime_config.sigmas[t]
                latents = latents + noise * dt

                Callbacks.in_loop(
                    t=t,
                    seed=seed,
                    prompt=prompt,
                    latents=latents,
                    config=runtime_config,
                    time_steps=time_steps,
                )

                mx.eval(latents)

            except KeyboardInterrupt:
                from mflux.error.exceptions import StopImageGenerationException

                Callbacks.interruption(
                    t=t,
                    seed=seed,
                    prompt=prompt,
                    latents=latents,
                    config=runtime_config,
                    time_steps=time_steps,
                )
                raise StopImageGenerationException(f"Stopping image generation at step {t + 1}/{len(time_steps)}")

        Callbacks.after_loop(
            seed=seed,
            prompt=prompt,
            latents=latents,
            config=runtime_config,
        )

        denoising_time = time.time() - denoising_start
        print(f"✅ Denoising completed in {denoising_time:.2f} seconds")

        print("🎨 Decoding latents to image...")
        decode_start = time.time()
        unpacked_latents = ArrayUtil.unpack_latents(
            latents=latents, height=runtime_config.height, width=runtime_config.width
        )

        unpacked_latents = unpacked_latents.reshape(
            unpacked_latents.shape[0],
            unpacked_latents.shape[1],
            1,
            unpacked_latents.shape[2],
            unpacked_latents.shape[3],
        )

        decoded = self.decode_latents(unpacked_latents)

        if len(decoded.shape) == 5:
            decoded = decoded[:, :, 0, :, :]

        decode_time = time.time() - decode_start
        print(f"✅ Decoding completed in {decode_time:.2f} seconds")

        total_time = time.time() - total_start
        print(f"\n🎯 Total generation time: {total_time:.2f} seconds")
        print(f"  - Denoising: {denoising_time:.2f}s ({denoising_time / total_time * 100:.1f}%)")
        print(f"  - Decoding: {decode_time:.2f}s ({decode_time / total_time * 100:.1f}%)")

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
    def from_name(model_name: str, quantize: int | None = None) -> "QwenImage":
        return QwenImage(
            model_config=ModelConfig.from_name(model_name=model_name),
            quantize=quantize,
        )

    def freeze(self, **kwargs):
        self.vae.freeze()
