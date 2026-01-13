"""Hunyuan-DiT transformer components.

Architecture Overview:
- 28 DiT blocks with self-attention + cross-attention + FFN
- 16 attention heads with 88 head dimension (1408 hidden dim)
- Cross-attention for text conditioning (CLIP + T5)
- AdaLN-Zero conditioning from timestep
- Rotary positional embeddings (RoPE)
"""

from mflux.models.hunyuan.model.hunyuan_transformer.hunyuan_dit import HunyuanDiT
from mflux.models.hunyuan.model.hunyuan_transformer.hunyuan_dit_block import HunyuanDiTBlock
from mflux.models.hunyuan.model.hunyuan_transformer.hunyuan_attention import HunyuanSelfAttention, HunyuanCrossAttention

__all__ = [
    "HunyuanDiT",
    "HunyuanDiTBlock",
    "HunyuanSelfAttention",
    "HunyuanCrossAttention",
]
