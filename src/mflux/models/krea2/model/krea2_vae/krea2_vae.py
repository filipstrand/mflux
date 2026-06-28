import mlx.core as mx

from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE


class Krea2VAE(QwenVAE):
    """Krea 2 reuses the Qwen-Image autoencoder (AutoencoderKLQwenImage, f8, 16 latent channels).

    The latents_mean / latents_std baked into QwenVAE match the Krea 2 vae config exactly, and the
    decode convention (`latents * std + mean`) equals diffusers' Krea2 decode (`latents / (1/std) + mean`),
    so QwenVAE is reused unchanged. Krea 2 unpacks latents to (B, 16, 1, H, W) before decode.
    """

    def decode(self, latents: mx.array) -> mx.array:
        # latents arrives as (B, 16, 1, H, W) from the Krea2 unpack.
        return super().decode(latents)
