"""NextDiT transformer for NewBie-image.

NextDiT is a Lumina-based diffusion transformer featuring:
- 36 transformer blocks with GQA attention
- AdaLN-Single modulation for conditioning
- 16-channel VAE latent space
- Patchified input embedding
- Learned position embeddings with RoPE

Architecture based on NewBie-image-Exp0.1 (3.5B parameters).
"""

import math

import mlx.core as mx
import mlx.nn as nn

from mflux.models.newbie.model.newbie_transformer.nextdit_block import NextDiTBlock


class NewBiePatchEmbed(nn.Module):
    """Patch embedding for NewBie-image.

    Converts latent image into patch tokens.

    Args:
        in_channels: Number of input channels (16 for NewBie's 16-ch VAE)
        hidden_dim: Hidden dimension (2560)
        patch_size: Patch size (2)
    """

    def __init__(
        self,
        in_channels: int = 16,
        hidden_dim: int = 2560,
        patch_size: int = 2,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        # Linear projection of flattened patches
        patch_dim = in_channels * patch_size * patch_size
        self.proj = nn.Linear(patch_dim, hidden_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Patchify and embed input.

        Args:
            x: Input latent [batch, channels, height, width]

        Returns:
            Patch embeddings [batch, num_patches, hidden_dim]
        """
        batch_size, channels, height, width = x.shape

        # Calculate number of patches
        h_patches = height // self.patch_size
        w_patches = width // self.patch_size

        # Reshape to patches: [batch, h_patches, patch_size, w_patches, patch_size, channels]
        x = x.reshape(
            batch_size,
            channels,
            h_patches,
            self.patch_size,
            w_patches,
            self.patch_size,
        )

        # Reorder: [batch, h_patches, w_patches, patch_size, patch_size, channels]
        x = x.transpose(0, 2, 4, 3, 5, 1)

        # Flatten patches: [batch, num_patches, patch_dim]
        x = x.reshape(batch_size, h_patches * w_patches, -1)

        # Project to hidden dimension
        x = self.proj(x)

        return x


class NewBieTimestepEmbed(nn.Module):
    """Timestep embedding for NewBie-image.

    Uses sinusoidal embeddings followed by MLP projection.

    Args:
        hidden_dim: Output dimension
        freq_dim: Frequency embedding dimension
    """

    def __init__(self, hidden_dim: int = 2560, freq_dim: int = 256):
        super().__init__()

        self.freq_dim = freq_dim

        # MLP to project frequency embeddings to hidden dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )

    def __call__(self, timesteps: mx.array) -> mx.array:
        """
        Embed timesteps.

        Args:
            timesteps: Timestep values [batch]

        Returns:
            Timestep embeddings [batch, hidden_dim]
        """
        # Sinusoidal position embedding
        half_dim = self.freq_dim // 2
        freqs = mx.exp(
            -math.log(10000) * mx.arange(0, half_dim, dtype=mx.float32) / half_dim
        )

        # Compute sin and cos embeddings
        args = timesteps[:, None] * freqs[None, :]
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)

        # Project through MLP
        return self.mlp(embedding)


class NewBieTextProjector(nn.Module):
    """Text embedding projector for NewBie-image.

    Projects and fuses embeddings from dual text encoders:
    - Gemma3 (primary): 2560 hidden dim
    - Jina CLIP (secondary): 1024 hidden dim

    Args:
        hidden_dim: Target hidden dimension (2560)
        gemma_dim: Gemma3 embedding dimension (2560)
        clip_dim: Jina CLIP embedding dimension (1024)
    """

    def __init__(
        self,
        hidden_dim: int = 2560,
        gemma_dim: int = 2560,
        clip_dim: int = 1024,
    ):
        super().__init__()

        # Gemma projection (identity if dims match)
        self.gemma_proj = (
            nn.Linear(gemma_dim, hidden_dim, bias=True)
            if gemma_dim != hidden_dim
            else None
        )

        # CLIP projection to match hidden dim
        self.clip_proj = nn.Linear(clip_dim, hidden_dim, bias=True)

        # Fusion layer (optional, for weighted combination)
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)

    def __call__(
        self,
        gemma_embeds: mx.array,
        clip_embeds: mx.array | None = None,
    ) -> mx.array:
        """
        Project and fuse text embeddings.

        Args:
            gemma_embeds: Gemma3 embeddings [batch, seq, gemma_dim]
            clip_embeds: Jina CLIP embeddings [batch, seq, clip_dim] (optional)

        Returns:
            Fused text embeddings [batch, seq, hidden_dim]
        """
        # Project Gemma embeddings
        if self.gemma_proj is not None:
            gemma_embeds = self.gemma_proj(gemma_embeds)

        # If no CLIP embeddings, just return Gemma
        if clip_embeds is None:
            return gemma_embeds

        # Project CLIP embeddings
        clip_embeds = self.clip_proj(clip_embeds)

        # Ensure sequence lengths match (pad CLIP if needed)
        gemma_seq = gemma_embeds.shape[1]
        clip_seq = clip_embeds.shape[1]

        if clip_seq < gemma_seq:
            # Pad CLIP with zeros
            padding = mx.zeros((clip_embeds.shape[0], gemma_seq - clip_seq, clip_embeds.shape[2]))
            clip_embeds = mx.concatenate([clip_embeds, padding], axis=1)
        elif clip_seq > gemma_seq:
            # Truncate CLIP
            clip_embeds = clip_embeds[:, :gemma_seq, :]

        # Fuse embeddings
        fused = mx.concatenate([gemma_embeds, clip_embeds], axis=-1)
        return self.fusion(fused)


class RoPEEmbedding(nn.Module):
    """Rotary Position Embedding for 2D image patches.

    Generates RoPE embeddings for patchified images.

    Args:
        head_dim: Per-head dimension
        max_seq_len: Maximum sequence length
        base: Base for frequency computation
    """

    def __init__(
        self,
        head_dim: int = 64,
        max_seq_len: int = 4096,
        base: float = 10000.0,
    ):
        super().__init__()

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (
            base ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim)
        )
        self.inv_freq = inv_freq

    def __call__(self, seq_len: int) -> tuple[mx.array, mx.array]:
        """
        Generate RoPE cos and sin for given sequence length.

        Args:
            seq_len: Sequence length

        Returns:
            Tuple of (cos, sin) each [seq_len, head_dim]
        """
        # Position indices
        positions = mx.arange(seq_len, dtype=mx.float32)

        # Compute angles
        freqs = mx.outer(positions, self.inv_freq)

        # Create full rotation matrix (interleaved cos/sin)
        emb = mx.concatenate([freqs, freqs], axis=-1)

        cos = mx.cos(emb)
        sin = mx.sin(emb)

        return cos, sin


class NextDiT(nn.Module):
    """NextDiT transformer for NewBie-image.

    Full transformer architecture with:
    - Patch embedding for 16-channel latents
    - Timestep and guidance conditioning
    - 36 NextDiT blocks with GQA
    - Text conditioning via cross-attention
    - Output projection to predict noise

    Args:
        model_config: Model configuration
        num_blocks: Number of transformer blocks (36)
        hidden_dim: Hidden dimension (2560)
        num_query_heads: Number of query heads (24)
        num_kv_heads: Number of KV heads for GQA (8)
        head_dim: Per-head dimension (64)
        mlp_dim: FFN intermediate dimension (6912)
        in_channels: Input latent channels (16)
        patch_size: Patch size (2)
        text_dim: Text embedding dimension (2560)
        qk_norm: Whether to use QK normalization
    """

    def __init__(
        self,
        model_config,
        num_blocks: int = 36,
        hidden_dim: int = 2560,
        num_query_heads: int = 24,
        num_kv_heads: int = 8,
        head_dim: int | None = None,
        mlp_dim: int = 6912,
        in_channels: int = 16,
        patch_size: int = 2,
        text_dim: int = 2560,
        qk_norm: bool = True,
    ):
        super().__init__()

        self.model_config = model_config
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        self.patch_size = patch_size

        head_dim = head_dim if head_dim is not None else hidden_dim // num_query_heads

        # Patch embedding
        self.patch_embed = NewBiePatchEmbed(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            patch_size=patch_size,
        )

        # Timestep embedding
        self.time_embed = NewBieTimestepEmbed(hidden_dim=hidden_dim)

        # Guidance embedding (separate from timestep for flexibility)
        self.guidance_embed = NewBieTimestepEmbed(hidden_dim=hidden_dim)

        # Text projector
        self.text_proj = NewBieTextProjector(
            hidden_dim=hidden_dim,
            gemma_dim=text_dim,
            clip_dim=1024,  # Jina CLIP dimension
        )

        # RoPE for positional encoding
        self.rope = RoPEEmbedding(head_dim=head_dim)

        # Transformer blocks
        self.blocks = [
            NextDiTBlock(
                hidden_dim=hidden_dim,
                num_query_heads=num_query_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                mlp_dim=mlp_dim,
                text_dim=hidden_dim,  # After projection
                has_cross_attention=True,
                qk_norm=qk_norm,
            )
            for _ in range(num_blocks)
        ]

        # Final normalization
        self.norm_out = nn.RMSNorm(hidden_dim, eps=1e-6)

        # Final AdaLN modulation for output
        self.final_adaLN = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)

        # Output projection (unpatchify)
        self.proj_out = nn.Linear(
            hidden_dim,
            patch_size * patch_size * in_channels,
            bias=True,
        )

    def __call__(
        self,
        latents: mx.array,
        timestep: mx.array,
        text_embeddings: mx.array,
        clip_embeddings: mx.array | None = None,
        guidance: mx.array | None = None,
    ) -> mx.array:
        """
        Forward pass for noise prediction.

        Args:
            latents: Noisy latents [batch, channels, height, width]
            timestep: Timestep [batch]
            text_embeddings: Gemma3 text embeddings [batch, seq, dim]
            clip_embeddings: Optional Jina CLIP embeddings [batch, seq, dim]
            guidance: Optional guidance scale [batch]

        Returns:
            Predicted noise [batch, channels, height, width]
        """
        batch_size, channels, height, width = latents.shape

        # Patchify latents
        hidden_states = self.patch_embed(latents)  # [batch, num_patches, hidden_dim]
        seq_len = hidden_states.shape[1]

        # Compute conditioning
        time_emb = self.time_embed(timestep)  # [batch, hidden_dim]

        if guidance is not None:
            guidance_emb = self.guidance_embed(guidance)
            conditioning = time_emb + guidance_emb
        else:
            conditioning = time_emb

        # Project text embeddings
        text_hidden = self.text_proj(text_embeddings, clip_embeddings)

        # Get RoPE embeddings
        rope_cos, rope_sin = self.rope(seq_len)

        # Process through transformer blocks
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                conditioning=conditioning,
                encoder_hidden_states=text_hidden,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )

        # Final normalization with AdaLN
        hidden_states = self.norm_out(hidden_states)

        # Apply final modulation
        final_mod = self.final_adaLN(conditioning)  # [batch, 2 * hidden_dim]
        scale, shift = mx.split(final_mod, 2, axis=-1)
        hidden_states = hidden_states * (1 + scale[:, None, :]) + shift[:, None, :]

        # Project to output
        output = self.proj_out(hidden_states)  # [batch, num_patches, patch_dim]

        # Unpatchify: reshape back to image
        h_patches = height // self.patch_size
        w_patches = width // self.patch_size

        # [batch, h_patches, w_patches, patch_size, patch_size, channels]
        output = output.reshape(
            batch_size,
            h_patches,
            w_patches,
            self.patch_size,
            self.patch_size,
            self.in_channels,
        )

        # Reorder to [batch, channels, height, width]
        output = output.transpose(0, 5, 1, 3, 2, 4)
        output = output.reshape(batch_size, self.in_channels, height, width)

        return output
