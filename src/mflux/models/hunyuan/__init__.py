"""Hunyuan-DiT model implementation.

Hunyuan-DiT is a diffusion transformer model with:
- 28 transformer blocks with self + cross attention
- 16 attention heads with 88 dim per head (1408 hidden dim)
- Dual text encoders: CLIP (1024 dim) + T5 (2048 dim)
- Standard 4-channel VAE
- DDPM scheduler (noise prediction)

Key architectural differences from FLUX:
- Uses separate cross-attention (not joint/MM attention)
- DDPM noise prediction vs Flow Matching velocity
- Rotary positional embeddings (RoPE)
- Bilingual support (English + Chinese)
"""
