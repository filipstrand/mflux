"""Hunyuan-DiT text encoder components.

Hunyuan-DiT uses dual text encoders:
1. CLIP encoder (1024 dim, 77 tokens) - Chinese CLIP model
2. T5/mT5 encoder (2048 dim, 256 tokens) - multilingual T5

Both encoders provide full sequence outputs (not just pooled),
which are projected and concatenated for cross-attention.
"""

from mflux.models.hunyuan.model.hunyuan_text_encoder.hunyuan_prompt_encoder import HunyuanPromptEncoder

__all__ = ["HunyuanPromptEncoder"]
