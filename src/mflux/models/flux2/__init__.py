"""FLUX.2 model implementation.

FLUX.2 is a 32B parameter flow-matching transformer model with:
- 8 joint transformer blocks + 48 single transformer blocks
- 48 attention heads with 128 dim per head (6144 hidden dim)
- Mistral3 multimodal text encoder (5120 hidden, GQA with 32 query/8 KV heads)
- 32-channel VAE (AutoencoderKLFlux2)
- Fused QKV+MLP projections in single blocks

Architecture differences from FLUX.1:
- Global modulation layers instead of per-block norm layers
- Fused to_qkv_mlp_proj in single blocks
- Different FFN structure (linear_in/linear_out vs net.0.proj/net.2)
- No text embedder (uses pooled Mistral3 output directly)
"""

from mflux.models.flux2.flux2_initializer import Flux2Initializer
from mflux.models.flux2.variants.txt2img.flux2 import Flux2

__all__ = ["Flux2", "Flux2Initializer"]
