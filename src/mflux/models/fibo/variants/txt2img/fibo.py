"""FIBO text-to-image model variant.

This is a minimal implementation that starts with just the VAE decoder.
We skip text encoding and transformer for now.
"""

from mlx import nn

from mflux.config.runtime_config import RuntimeConfig
from mflux.models.fibo.fibo_initializer import FIBOInitializer
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

    vae: nn.Module

    def __init__(
        self,
        quantize: int | None = None,
        local_path: str | None = None,
    ):
        """Initialize FIBO model.

        Args:
            quantize: Quantization bits (not implemented yet)
            local_path: Local model path (optional)
        """
        super().__init__()
        # Initialize model structure (VAE will be created by initializer)
        self.vae = None
        self.bits = quantize
        self.local_path = local_path

        # Load weights and initialize model components
        FIBOInitializer.init(
            fibo_model=self,
            quantize=quantize,
            local_path=local_path,
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        config: RuntimeConfig,
        negative_prompt: str | None = None,
    ) -> GeneratedImage:
        """Generate image from latents (VAE decode only for now).

        This is a minimal implementation that:
        1. Loads the exact scaled latents from PyTorch debug output (saved by debug_diffusers_txt2img.py)
        2. Decodes them using VAE
        3. Returns the image

        Args:
            seed: Random seed (not used yet, latents come from PyTorch)
            prompt: Text prompt (not used yet, skipping text encoding)
            config: Generation config
            negative_prompt: Negative prompt (not used yet)

        Returns:
            Generated image
        """
        # CRITICAL: Load the exact tensor saved by diffusers pipeline right before VAE decode
        # This ensures we're using the exact same input, making debugging much easier
        # The diffusers pipeline saves "vae_input_latents" after scaling: (latents / latents_std) + latents_mean
        # So we use that directly - no need to scale again
        # debug_load() already returns an MLX array, so no conversion needed
        latents_for_vae = debug_load("vae_input_latents")

        # Ensure 5D shape for VAE: (batch, channels, 1, height, width)
        # The diffusers pipeline saves it as 5D, but check just in case
        if latents_for_vae.ndim == 4:
            latents_for_vae = latents_for_vae.reshape(
                latents_for_vae.shape[0],
                latents_for_vae.shape[1],
                1,
                latents_for_vae.shape[2],
                latents_for_vae.shape[3],
            )

        # Decode using VAE with the exact tensor from diffusers
        decoded = self.vae.decode(latents_for_vae)

        # Note: unpatchify already reduced channels from 12 to 3, so decoded is already RGB
        # Convert to image and return
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
