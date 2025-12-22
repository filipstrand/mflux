from pathlib import Path
from typing import List

import mlx.core as mx
import numpy as np
from mlx import nn
from PIL import Image

from mflux.models.common.config import ModelConfig
from mflux.models.common.config.config import Config
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.qwen.model.qwen_text_encoder.qwen_prompt_encoder import QwenPromptEncoder
from mflux.models.qwen.model.qwen_text_encoder.qwen_text_encoder import QwenTextEncoder
from mflux.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer  # Use base transformer!
from mflux.models.qwen_layered.model.qwen_layered_vae.qwen_layered_vae import QwenLayeredVAE
from mflux.models.qwen_layered.qwen_layered_initializer import QwenLayeredInitializer
from mflux.models.qwen_layered.weights.qwen_layered_weight_definition import QwenLayeredWeightDefinition
from mflux.utils.exceptions import StopImageGenerationException


class QwenImageLayered(nn.Module):
    """
    Qwen-Image-Layered model for image decomposition into RGBA layers.

    Takes an input RGB image and decomposes it into N semantically disentangled
    RGBA layers that can be independently edited and composited back together.
    """

    vae: QwenLayeredVAE
    transformer: QwenTransformer  # Use base Qwen transformer!
    text_encoder: QwenTextEncoder

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        model_config: ModelConfig = None,
    ):
        super().__init__()
        if model_config is None:
            model_config = ModelConfig.from_name("qwen-image-layered")

        QwenLayeredInitializer.init(
            model=self,
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            model_config=model_config,
        )

    def decompose(
        self,
        seed: int,
        image_path: Path | str,
        num_layers: int = 4,
        num_inference_steps: int = 50,
        guidance: float = 4.0,
        resolution: int = 640,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        scheduler: str = "linear",
        cfg_normalize: bool = True,
    ) -> List[Image.Image]:
        """
        Decompose an input image into N RGBA layers.
        """
        # Load and preprocess input image
        input_image = Image.open(image_path).convert("RGBA")

        # Resize to target resolution while maintaining aspect ratio
        width, height = self._compute_resolution(input_image.size, resolution)
        input_image = input_image.resize((width, height), Image.Resampling.LANCZOS)

        # Create config
        config = Config(
            width=width,
            height=height,
            guidance=guidance,
            scheduler=scheduler,
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
        )

        # Encode input image to latent (RGB only)
        input_tensor = self._image_to_tensor(input_image)
        # Encode only RGB (first 3 channels) -> [B, 16, H/8, W/8]
        cond_latent = self.vae.encode_condition_image(input_tensor[:, :3, :, :])

        # Add layer dimension: [B, C, H, W] -> [B, C, 1, H, W] -> [B, 1, C, H, W]
        cond_latent = mx.expand_dims(cond_latent, axis=2)  # [B, C, 1, H, W]
        cond_latent = cond_latent.transpose(0, 2, 1, 3, 4)  # [B, 1, C, H, W]

        # Pack condition image latent
        image_latents = self._pack_latents(
            cond_latent,
            batch_size=1,
            num_layers=1,  # Single condition image
            height=height,
            width=width,
        )

        # Create initial noise for output layers (layers+1 to include combined)
        mx.random.seed(seed)
        noise = self._create_noise(
            seed=seed,
            num_layers=num_layers + 1,  # +1 for combined output
            height=height,
            width=width,
        )
        latents = self._pack_latents(
            noise,
            batch_size=1,
            num_layers=num_layers + 1,
            height=height,
            width=width,
        )

        # Encode prompt
        if prompt is None or prompt == "":
            prompt = "an image"  # Placeholder - ideally auto-caption
        if negative_prompt is None:
            negative_prompt = "blurry, bad quality"

        prompt_embeds, prompt_mask, neg_embeds, neg_mask = QwenPromptEncoder.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_cache=self.prompt_cache,
            qwen_tokenizer=self.tokenizers["qwen"],
            qwen_text_encoder=self.text_encoder,
        )

        # Calculate latent dimensions and img_shapes for RoPE
        latent_height = height // 16  # VAE compression / patch
        latent_width = width // 16

        # Build cond_image_grid: (num_layers+1 output) + 1 condition
        # Each layer is: (1 frame, latent_height, latent_width)
        # For layered model, we need to generate RoPE for ALL layers + condition
        # The base transformer will compute shapes for: [(1, H, W)] + cond_image_grid
        # So we pass (num_layers+1 - 1) additional grids for the noisy layers,
        # plus 1 for condition = num_layers + 1 additional grids
        cond_image_grid = [(1, latent_height, latent_width) for _ in range(num_layers + 1)]

        # Denoising loop
        print(f"Decomposing into {num_layers} layers...")
        try:
            for step_idx, t in enumerate(config.time_steps):
                # Scale model input
                latents = config.scheduler.scale_model_input(latents, t)

                # KEY DIFFERENCE: Concatenate noisy latents with condition image
                # Like Diffusers: latent_model_input = torch.cat([latents, image_latents], dim=1)
                latent_model_input = mx.concatenate([latents, image_latents], axis=1)

                # Predict noise with positive prompt using BASE transformer
                noise_pred = self.transformer(
                    t=t,
                    config=config,
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_mask,
                    cond_image_grid=cond_image_grid,
                )
                # Only take the first part (excludes condition image)
                noise_pred = noise_pred[:, : latents.shape[1], :]

                # Predict noise with negative prompt
                noise_pred_neg = self.transformer(
                    t=t,
                    config=config,
                    hidden_states=mx.concatenate([latents, image_latents], axis=1),
                    encoder_hidden_states=neg_embeds,
                    encoder_hidden_states_mask=neg_mask,
                    cond_image_grid=cond_image_grid,
                )
                noise_pred_neg = noise_pred_neg[:, : latents.shape[1], :]

                # Apply CFG
                guided_noise = self._compute_guided_noise(noise_pred, noise_pred_neg, guidance, cfg_normalize)

                # Scheduler step
                latents = config.scheduler.step(noise=guided_noise, timestep=t, latents=latents)

                # Evaluate for progress
                mx.eval(latents)

                if (step_idx + 1) % 10 == 0 or step_idx == 0:
                    print(f"  Step {step_idx + 1}/{num_inference_steps}")
        except KeyboardInterrupt:
            raise StopImageGenerationException(
                f"Stopping decomposition at step {step_idx + 1}/{num_inference_steps}"
            ) from None

        # Unpack latents
        latents = self._unpack_latents(
            latents,
            num_layers=num_layers + 1,  # +1 for combined
            height=height,
            width=width,
        )

        # Skip first frame (combined image) - like Diffusers line 886
        # latents[:, :, 1:] - skip first layer
        latents = latents[:, 1:, :, :, :]  # Shape: [B, layers, C, H, W]

        # Decode each layer
        output_images = []
        print("  Decoding layers...")
        for layer_idx in range(num_layers):
            layer_latent = latents[:, layer_idx : layer_idx + 1, :, :, :]  # [B, 1, C, H, W]
            layer_latent = layer_latent[:, 0, :, :, :]  # [B, C, H, W]
            decoded = self.vae.decode(layer_latent, num_layers=1)
            rgba_image = self._tensor_to_image(decoded)
            output_images.append(rgba_image)

        print(f"Decomposition complete: {num_layers} layers")
        return output_images

    def _compute_resolution(self, original_size: tuple, target_bucket: int) -> tuple:
        """Compute target resolution maintaining aspect ratio."""
        w, h = original_size
        aspect = w / h

        if aspect >= 1.0:
            # Landscape or square
            new_w = target_bucket
            new_h = int(target_bucket / aspect)
        else:
            # Portrait
            new_h = target_bucket
            new_w = int(target_bucket * aspect)

        # Round to nearest multiple of 16 for VAE compatibility
        new_w = (new_w // 16) * 16
        new_h = (new_h // 16) * 16

        return max(new_w, 16), max(new_h, 16)

    def _create_noise(self, seed: int, num_layers: int, height: int, width: int) -> mx.array:
        """Create initial noise for output layers."""
        latent_height = height // 8  # VAE compression
        latent_width = width // 8
        num_channels = 16  # Qwen VAE latent channels

        mx.random.seed(seed)
        noise = mx.random.normal(shape=(1, num_layers, num_channels, latent_height, latent_width))
        return noise.astype(mx.bfloat16)

    def _pack_latents(
        self,
        latents: mx.array,
        batch_size: int,
        num_layers: int,
        height: int,
        width: int,
    ) -> mx.array:
        """
        Pack latents for transformer input.

        From Diffusers:
        latents = latents.view(batch_size, layers, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4, 6)
        latents = latents.reshape(batch_size, layers * (height // 2) * (width // 2), num_channels_latents * 4)
        """
        # latents: [B, layers, C, H, W]
        latent_height = height // 8 // 2  # VAE + patch
        latent_width = width // 8 // 2
        num_channels = latents.shape[2]

        # Reshape: [B, layers, C, H/2, 2, W/2, 2]
        latents = latents.reshape(batch_size, num_layers, num_channels, latent_height, 2, latent_width, 2)

        # Permute: [B, layers, H/2, W/2, C, 2, 2]
        latents = latents.transpose(0, 1, 3, 5, 2, 4, 6)

        # Reshape: [B, layers * H/2 * W/2, C * 4]
        latents = latents.reshape(batch_size, num_layers * latent_height * latent_width, num_channels * 4)

        return latents

    def _unpack_latents(
        self,
        latents: mx.array,
        num_layers: int,
        height: int,
        width: int,
    ) -> mx.array:
        """
        Unpack latents after transformer.

        From Diffusers:
        latents = latents.view(batch_size, layers + 1, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 1, 4, 2, 5, 3, 6)
        latents = latents.reshape(batch_size, layers + 1, channels // (2 * 2), height, width)
        """
        batch_size = latents.shape[0]
        channels = latents.shape[2]

        latent_height = height // 8 // 2  # VAE + patch
        latent_width = width // 8 // 2

        # Reshape: [B, layers, H/2, W/2, C/4, 2, 2]
        latents = latents.reshape(batch_size, num_layers, latent_height, latent_width, channels // 4, 2, 2)

        # Permute: [B, layers, C/4, H/2, 2, W/2, 2]
        latents = latents.transpose(0, 1, 4, 2, 5, 3, 6)

        # Reshape: [B, layers, C, H, W]
        full_height = latent_height * 2
        full_width = latent_width * 2
        latents = latents.reshape(batch_size, num_layers, channels // 4, full_height, full_width)

        return latents

    def _image_to_tensor(self, image: Image.Image) -> mx.array:
        """Convert PIL RGBA image to tensor [1, 4, H, W] in [-1, 1]."""
        arr = np.array(image).astype(np.float32) / 255.0
        arr = arr * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        arr = np.transpose(arr, (2, 0, 1))  # [H, W, C] -> [C, H, W]
        arr = np.expand_dims(arr, 0)  # [1, C, H, W]
        return mx.array(arr)

    def _tensor_to_image(self, tensor: mx.array) -> Image.Image:
        """Convert tensor [1, 4, H, W] in [-1, 1] to PIL RGBA image."""
        arr = np.array(tensor[0])  # [4, H, W]
        arr = np.transpose(arr, (1, 2, 0))  # [H, W, 4]
        arr = (arr + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGBA")

    @staticmethod
    def _compute_guided_noise(
        noise: mx.array,
        noise_neg: mx.array,
        guidance: float,
        normalize: bool = True,
    ) -> mx.array:
        """Apply classifier-free guidance with optional normalization."""
        combined = noise_neg + guidance * (noise - noise_neg)

        if normalize:
            cond_norm = mx.sqrt(mx.sum(noise * noise, axis=-1, keepdims=True) + 1e-12)
            combined_norm = mx.sqrt(mx.sum(combined * combined, axis=-1, keepdims=True) + 1e-12)
            combined = combined * (cond_norm / combined_norm)

        return combined

    def save_model(self, base_path: str) -> None:
        """Save the model with current quantization."""
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=QwenLayeredWeightDefinition,
        )
