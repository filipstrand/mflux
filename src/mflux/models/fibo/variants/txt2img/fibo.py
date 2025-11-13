"""FIBO text-to-image model variant.

This is a minimal implementation that starts with just the VAE decoder.
We skip text encoding and transformer for now.
"""

import mlx.core as mx
from mlx import nn

from mflux.config.config import Config
from mflux.models.fibo.model.fibo_vae.vae import VAE
from mflux.post_processing.generated_image import GeneratedImage
from mflux.post_processing.image_util import ImageUtil
from mflux_debugger.tensor_debug import debug_load


class FIBO(nn.Module):
    """FIBO model - minimal implementation starting with VAE decoder only.

    For now, we skip:
    - Text encoding (T5, CLIP)
    - Transformer (denoising)

    We only implement:
    - VAE decoder (to decode latents from PyTorch)
    """

    vae: VAE

    def __init__(
        self,
        quantize: int | None = None,
        local_path: str | None = None,
    ):
        """Initialize FIBO model.

        Args:
            quantize: Quantization bits (not implemented yet)
            local_path: Local model path (not implemented yet)
        """
        super().__init__()
        # For now, just initialize VAE
        self.vae = VAE()
        self.bits = quantize
        self.local_path = local_path
        # TODO: Load weights once we implement weight loading

    def generate_image(
        self,
        seed: int,
        prompt: str,
        config: Config,
        negative_prompt: str | None = None,
    ) -> GeneratedImage:
        """Generate image from latents (VAE decode only for now).

        This is a minimal implementation that:
        1. Loads latents from PyTorch debug output (saved by debug_diffusers_txt2img.py)
        2. Scales latents: (latents / latents_std) + latents_mean
        3. Decodes them using VAE
        4. Returns the image

        Args:
            seed: Random seed (not used yet, latents come from PyTorch)
            prompt: Text prompt (not used yet, skipping text encoding)
            config: Generation config
            negative_prompt: Negative prompt (not used yet)

        Returns:
            Generated image
        """
        # Load latents from PyTorch debug output
        # The latents from PyTorch are saved as "vae_input_latents"
        # Shape should be (batch, 48, 1, height, width) for FIBO
        latents_pytorch = debug_load("vae_input_latents")

        # Convert to MLX array
        latents_mlx = mx.array(latents_pytorch)

        # Scale latents: (latents / latents_std) + latents_mean
        # latents_mean and latents_std are per-channel (48 channels)
        # We need to reshape them to broadcast correctly: (1, 48, 1, 1, 1) for 5D latents
        latents_mean_mlx = mx.array(VAE.LATENTS_MEAN)
        latents_std_mlx = mx.array(VAE.LATENTS_STD)

        # Ensure latents are 5D: (batch, channels, 1, height, width)
        if latents_mlx.ndim == 4:
            latents_mlx = latents_mlx.reshape(
                latents_mlx.shape[0], latents_mlx.shape[1], 1, latents_mlx.shape[2], latents_mlx.shape[3]
            )

        # Reshape mean/std for broadcasting: (1, 48, 1, 1, 1)
        latents_mean_mlx = latents_mean_mlx.reshape(1, -1, 1, 1, 1)
        latents_std_mlx = latents_std_mlx.reshape(1, -1, 1, 1, 1)

        # Scale: (latents / latents_std) + latents_mean
        latents_scaled = (latents_mlx / latents_std_mlx) + latents_mean_mlx

        # Decode using VAE
        decoded = self.vae.decode(latents_scaled)

        # Convert to image and return
        # Note: decoded is (batch, 12, height, width) - FIBO outputs 12 channels
        # ImageUtil.to_image expects (batch, channels, height, width)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            lora_paths=None,
            lora_scales=None,
            image_path=None,
            image_strength=None,
            generation_time=0.0,  # Placeholder
        )
