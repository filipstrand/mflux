"""NewBie-image text encoder components.

NewBie uses dual text encoders:
- Gemma3-4B-it: Primary text encoder (2560 hidden dim)
- Jina CLIP v2: Secondary encoder for semantic alignment (1024 hidden dim)
"""

from mflux.models.newbie.model.newbie_text_encoder.gemma3_encoder import Gemma3Encoder
from mflux.models.newbie.model.newbie_text_encoder.jina_clip_encoder import JinaCLIPEncoder

__all__ = ["Gemma3Encoder", "JinaCLIPEncoder"]
