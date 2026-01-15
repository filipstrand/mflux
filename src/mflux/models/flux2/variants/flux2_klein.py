import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import PIL.Image
from mlx import nn

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux2.flux2_initializer import Flux2Initializer
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
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
            latent_ids = self._prepare_latent_ids_from_packed(latents)
        else:
            latents, latent_ids, latent_height, latent_width = self._prepare_latents(
                seed=seed,
                height=height,
                width=width,
                batch_size=1,
            )
            latents = self._pack_latents(latents)

        # 3. Prepare timesteps and sigmas
        image_seq_len = latents.shape[1]
        timesteps, sigmas = self._get_timesteps_and_sigmas(
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
        prompt_embeds = self._get_qwen3_prompt_embeds(
            prompt=prompt,
            max_sequence_length=max_sequence_length,
            hidden_state_layers=text_encoder_out_layers,
        )
        if num_images_per_prompt > 1:
            prompt_embeds = mx.repeat(prompt_embeds, num_images_per_prompt, axis=0)
        text_ids = self._prepare_text_ids(prompt_embeds)
        return prompt_embeds, text_ids

    def _get_qwen3_prompt_embeds(
        self,
        prompt: str | list[str],
        max_sequence_length: int,
        hidden_state_layers: tuple[int, ...],
    ) -> mx.array:
        tokenizer = self.tokenizers["qwen3"]
        tokens = tokenizer.tokenize(prompt=prompt, max_length=max_sequence_length)
        return self.text_encoder.get_prompt_embeds(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
            hidden_state_layers=hidden_state_layers,
        )

    @staticmethod
    def _prepare_text_ids(x: mx.array, t_coord: mx.array | None = None) -> mx.array:
        batch_size, seq_len, _ = x.shape
        out_ids = []
        for i in range(batch_size):
            if t_coord is None:
                t = mx.zeros((seq_len,), dtype=mx.int32)
            else:
                t = t_coord[i]
                if t.ndim == 0:
                    t = mx.full((seq_len,), t, dtype=mx.int32)
                elif t.shape[0] != seq_len:
                    t = mx.broadcast_to(t, (seq_len,))
                t = t.astype(mx.int32)
            h = mx.zeros((seq_len,), dtype=mx.int32)
            w = mx.zeros((seq_len,), dtype=mx.int32)
            token_ids = mx.arange(seq_len, dtype=mx.int32)
            coords = mx.stack([t, h, w, token_ids], axis=1)
            out_ids.append(coords)
        return mx.stack(out_ids, axis=0)

    @staticmethod
    def _pack_latents(latents: mx.array) -> mx.array:
        batch_size, num_channels, height, width = latents.shape
        return latents.reshape(batch_size, num_channels, height * width).transpose(0, 2, 1)

    @staticmethod
    def _prepare_latent_ids(latents: mx.array) -> mx.array:
        batch_size, _, height, width = latents.shape
        h_ids = mx.arange(height, dtype=mx.int32)
        w_ids = mx.arange(width, dtype=mx.int32)
        h_grid = mx.broadcast_to(mx.expand_dims(h_ids, axis=1), (height, width))
        w_grid = mx.broadcast_to(mx.expand_dims(w_ids, axis=0), (height, width))
        flat_h = h_grid.reshape(-1)
        flat_w = w_grid.reshape(-1)
        t = mx.zeros_like(flat_h)
        layer_ids = mx.zeros_like(flat_h)
        coords = mx.stack([t, flat_h, flat_w, layer_ids], axis=1)
        coords = mx.expand_dims(coords, axis=0)
        return mx.broadcast_to(coords, (batch_size, coords.shape[1], coords.shape[2]))

    @staticmethod
    def _prepare_latents(
        seed: int,
        height: int,
        width: int,
        batch_size: int,
        num_latents_channels: int = 32,
        vae_scale_factor: int = 8,
    ) -> tuple[mx.array, mx.array, int, int]:
        height = 2 * (height // (vae_scale_factor * 2))
        width = 2 * (width // (vae_scale_factor * 2))
        latent_height = height // 2
        latent_width = width // 2
        latents = mx.random.normal(
            shape=(batch_size, num_latents_channels * 4, latent_height, latent_width),
            key=mx.random.key(seed),
        ).astype(ModelConfig.precision)
        latent_ids = Flux2Klein._prepare_latent_ids(latents)
        return latents, latent_ids, latent_height, latent_width

    @staticmethod
    def _prepare_latent_ids_from_packed(latents: mx.array) -> mx.array:
        batch_size, seq_len, _ = latents.shape
        height = int(mx.sqrt(mx.array(seq_len, dtype=mx.float32)).item())
        width = seq_len // height if height > 0 else 0
        h_ids = mx.arange(height, dtype=mx.int32)
        w_ids = mx.arange(width, dtype=mx.int32)
        h_grid = mx.broadcast_to(mx.expand_dims(h_ids, axis=1), (height, width))
        w_grid = mx.broadcast_to(mx.expand_dims(w_ids, axis=0), (height, width))
        flat_h = h_grid.reshape(-1)
        flat_w = w_grid.reshape(-1)
        t = mx.zeros_like(flat_h)
        layer_ids = mx.zeros_like(flat_h)
        coords = mx.stack([t, flat_h, flat_w, layer_ids], axis=1)
        coords = mx.expand_dims(coords, axis=0)
        return mx.broadcast_to(coords, (batch_size, coords.shape[1], coords.shape[2]))

    @staticmethod
    def _compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
        a1, b1 = 8.73809524e-05, 1.89833333
        a2, b2 = 0.00016927, 0.45666666
        if image_seq_len > 4300:
            return float(a2 * image_seq_len + b2)
        m_200 = a2 * image_seq_len + b2
        m_10 = a1 * image_seq_len + b1
        a = (m_200 - m_10) / 190.0
        b = m_200 - 200.0 * a
        return float(a * num_steps + b)

    @staticmethod
    def _time_shift_exponential(mu: float, sigma_power: float, t: mx.array) -> mx.array:
        return mx.exp(mu) / (mx.exp(mu) + ((1.0 / t - 1.0) ** sigma_power))

    @staticmethod
    def _get_timesteps_and_sigmas(
        image_seq_len: int,
        num_inference_steps: int,
        num_train_timesteps: int = 1000,
    ) -> tuple[mx.array, mx.array]:
        sigmas = mx.array(np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps), dtype=mx.float32)
        mu = Flux2Klein._compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
        sigmas = Flux2Klein._time_shift_exponential(mu, 1.0, sigmas)
        timesteps = sigmas * num_train_timesteps
        sigmas = mx.concatenate([sigmas, mx.zeros((1,), dtype=sigmas.dtype)], axis=0)
        return timesteps, sigmas

    def debug_decode_packed_latents(
        self,
        latents_path: str | Path,
        output_path: str | Path | None = None,
    ) -> PIL.Image.Image:
        latents_path = Path(latents_path)
        data = mx.load(str(latents_path))
        if isinstance(data, dict):
            packed_latents = next(iter(data.values()))
        else:
            packed_latents = data

        latents = mx.array(packed_latents)
        if latents.ndim >= 4 and latents.shape[1] == self.vae.latent_channels:
            decoded = self.vae.decode(latents)
        else:
            decoded = self.vae.decode_packed_latents(latents)
        normalized = ImageUtil._denormalize(decoded)
        image = ImageUtil._numpy_to_pil(ImageUtil._to_numpy(normalized))
        if output_path is not None:
            image.save(output_path)
        return image

    def debug_roundtrip_image(
        self,
        image_path: str | Path,
        output_path: str | Path | None = None,
    ) -> PIL.Image.Image:
        image = ImageUtil.load_image(image_path)
        image_array = ImageUtil.to_array(image)
        latents = self.vae.encode(image_array)
        decoded = self.vae.decode(latents)
        normalized = ImageUtil._denormalize(decoded)
        out_image = ImageUtil._numpy_to_pil(ImageUtil._to_numpy(normalized))
        if output_path is not None:
            out_image.save(output_path)
        return out_image

    def debug_transformer_step(
        self,
        inputs_path: str | Path,
        output_path: str | Path | None = None,
    ) -> dict[str, float]:
        data = np.load(Path(inputs_path))
        latents = mx.array(data["latents"])
        latent_ids = mx.array(data["latent_ids"])
        prompt_embeds = mx.array(data["prompt_embeds"])
        text_ids = mx.array(data["text_ids"])
        timestep = mx.array(data["timestep"])

        noise_pred = self.transformer(
            hidden_states=latents,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep / 1000,
            img_ids=latent_ids,
            txt_ids=text_ids,
            guidance=None,
        )

        if output_path is not None:
            np.save(Path(output_path), np.array(noise_pred))

        metrics: dict[str, float] = {}
        if "noise_pred" in data:
            ref = mx.array(data["noise_pred"])
            diff = mx.abs(noise_pred - ref)
            metrics["mean_abs_error"] = float(mx.mean(diff))
            metrics["max_abs_error"] = float(mx.max(diff))
        return metrics

    def debug_text_encoder(
        self,
        prompt: str | list[str],
        inputs_path: str | Path,
        max_sequence_length: int = 512,
        text_encoder_out_layers: tuple[int, ...] = (9, 18, 27),
    ) -> dict[str, float]:
        data = np.load(Path(inputs_path))
        ref_prompt_embeds = mx.array(data["prompt_embeds"])
        ref_text_ids = mx.array(data["text_ids"])

        prompt_embeds = self._get_qwen3_prompt_embeds(
            prompt=prompt,
            max_sequence_length=max_sequence_length,
            hidden_state_layers=text_encoder_out_layers,
        )
        text_ids = self._prepare_text_ids(prompt_embeds)

        metrics: dict[str, float] = {}
        diff = mx.abs(prompt_embeds - ref_prompt_embeds)
        metrics["mean_abs_error"] = float(mx.mean(diff))
        metrics["max_abs_error"] = float(mx.max(diff))
        metrics["text_ids_match"] = float(mx.mean((text_ids == ref_text_ids).astype(mx.float32)))
        return metrics
