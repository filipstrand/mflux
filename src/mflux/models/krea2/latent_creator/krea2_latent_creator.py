"""Latent helpers for Krea-2.

Krea-2 keeps latents as unpacked ``(B, 16, h, w)`` throughout the loop (the DiT
patchifies/unpatchifies internally and the Qwen-Image VAE decodes ``(B,16,h,w)``
directly), so ``unpack_latents`` is the identity — it exists only to satisfy the
stepwise-handler / callback interface.
"""

import mlx.core as mx


class Krea2LatentCreator:
    @staticmethod
    def unpack_latents(latents: mx.array, height: int, width: int) -> mx.array:
        return latents
