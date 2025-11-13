"""FIBO model initializer.

Loads weights and initializes the FIBO model components.
"""

import mlx.core as mx

from mflux.models.fibo.model.fibo_vae.vae import VAE
from mflux.models.fibo.weights.fibo_weight_handler import FIBOWeightHandler


class FIBOInitializer:
    """Initializer for FIBO model."""

    @staticmethod
    def init(
        fibo_model,
        quantize: int | None = None,
        local_path: str | None = None,
    ) -> None:
        """Initialize FIBO model with weights.

        Args:
            fibo_model: FIBO model instance to initialize
            quantize: Quantization bits (not implemented yet)
            local_path: Local path to model weights (optional)
        """
        # 1. Load VAE weights
        weights = FIBOWeightHandler.load_regular_weights(
            repo_id="briaai/FIBO",  # Default FIBO model
            local_path=local_path,
        )

        # 2. Initialize VAE model
        fibo_model.vae = VAE()

        # 3. Apply weights to VAE
        # The weights from FIBOWeightHandler are already in the correct nested MLX structure
        # (including resample Conv2d weights via FIBOWeightMapping). MLX's model.update()
        # accepts nested dictionaries directly.
        if weights.vae:
            fibo_model.vae.update(weights.vae, strict=False)

        # 4. Store quantization level (not implemented yet)
        fibo_model.bits = quantize
        fibo_model.local_path = local_path

        # 5. Optionally quantize the model (not implemented yet)
        if quantize:
            # TODO: Implement quantization for FIBO VAE
            pass


def _force_resample_conv_from_diffusers(fibo_model, local_path: str | None) -> None:
    """Overwrite decoder upsample resample Conv2d weights from diffusers BriaFiboPipeline.

    This is a targeted fix for the WanResample Conv2d inside decoder.up_blocks.{block}.upsampler:
    at runtime we've observed that MLX's resample_conv weights do not numerically match the
    PyTorch decoder.up_blocks.{block}.upsampler.resample.1.weight used by BriaFiboPipeline.

    By reloading those specific Conv2d weights and bias directly from diffusers and applying
    the same transpose used in FIBOWeightMapping, we ensure the MLX decoder's resample convs
    are bit-for-bit aligned with the PyTorch implementation.
    """

    # Local imports to avoid hard dependencies if this helper is unused in other contexts.
    import torch
    from diffusers import BriaFiboPipeline

    from mflux.models.fibo.weights.fibo_weight_mapping import transpose_conv2d_weight

    # Load the same pipeline the debugger uses.
    pipe = BriaFiboPipeline.from_pretrained(local_path or "briaai/FIBO", torch_dtype=torch.bfloat16)

    # There are up to 4 decoder up_blocks; only blocks with an upsampler
    # (0, 1, 2 in the FIBO config) have the resample Conv2d we care about.
    for block_idx, up_block_mlx in enumerate(fibo_model.vae.decoder.up_blocks):
        upsampler_mlx = getattr(up_block_mlx, "upsampler", None)
        if upsampler_mlx is None:
            continue

        # PyTorch side: historically this was a Sequential[WanUpsample, Conv2d].
        # Newer diffusers code may expose a WanResample module directly.
        up_block_pt = pipe.vae.decoder.up_blocks[block_idx]
        resample_mod = getattr(up_block_pt, "upsampler", None)
        if resample_mod is None:
            continue

        # Try to locate the Conv2d inside the PyTorch upsampler in a robust way.
        # 1) Sequential case: index 1.
        conv2d_pt = None
        if hasattr(resample_mod, "__len__") and len(resample_mod) >= 2:
            conv2d_pt = resample_mod[1]
        else:
            # 2) Monolithic module (e.g. WanResample) – look for a Conv2d attribute.
            import torch.nn as nn  # type: ignore

            for attr_name in dir(resample_mod):
                maybe = getattr(resample_mod, attr_name, None)
                if isinstance(maybe, nn.Conv2d):
                    conv2d_pt = maybe
                    break

        if conv2d_pt is None:
            # If we can't confidently find the Conv2d, skip overriding for this block.
            continue

        # Extract weights/bias from PyTorch, convert to MLX layout.
        w_pt = conv2d_pt.weight.to(torch.float32).detach().cpu().numpy()
        b_pt = conv2d_pt.bias.to(torch.float32).detach().cpu().numpy()

        w_mlx = transpose_conv2d_weight(mx.array(w_pt))
        b_mlx = mx.array(b_pt)

        # Overwrite MLX resample_conv in-place.
        upsampler_mlx.resample_conv.weight = w_mlx
        upsampler_mlx.resample_conv.bias = b_mlx
