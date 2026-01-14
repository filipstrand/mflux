"""Hunyuan-DiT Main Transformer Model.

HunyuanDiT2DModel is the main diffusion transformer for Hunyuan.

Architecture:
- 28 DiT transformer blocks
- 16 attention heads, 88 head dim (1408 hidden dim)
- Text encoder: T5 (2048 dim) projected to 1024
- Standard 4-channel VAE latent space
- AdaLN-Zero timestep conditioning (on norm1 only)
- Rotary position embeddings (RoPE)

Input:
- Noisy latents from VAE [batch, 4, H/8, W/8]
- Text embeddings from T5 [batch, 256, 2048]
- Timestep [batch]

Output:
- Predicted noise [batch, 4, H/8, W/8]
"""

import math

import mlx.core as mx
from mlx import nn

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.hunyuan.model.hunyuan_transformer.hunyuan_dit_block import HunyuanDiTBlock


class HunyuanTimestepEmbed(nn.Module):
    """Timestep embedding for Hunyuan-DiT.

    Converts scalar timestep to embedding vector using sinusoidal encoding
    followed by MLP projection.
    """

    def __init__(
        self,
        hidden_dim: int = 1408,
        frequency_embedding_size: int = 256,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.frequency_embedding_size = frequency_embedding_size

        # MLP to project sinusoidal encoding
        self.linear_1 = nn.Linear(frequency_embedding_size, hidden_dim, bias=True)
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def __call__(self, timesteps: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            timesteps: Timestep values [batch] or scalar

        Returns:
            Timestep embeddings [batch, hidden_dim]
        """
        # Ensure timesteps is 1D
        if timesteps.ndim == 0:
            timesteps = mx.expand_dims(timesteps, axis=0)

        # Sinusoidal encoding
        t_emb = self._timestep_embedding(timesteps, self.frequency_embedding_size)

        # MLP projection
        t_emb = self.linear_1(t_emb)
        t_emb = nn.silu(t_emb)
        t_emb = self.linear_2(t_emb)

        return t_emb

    @staticmethod
    def _timestep_embedding(
        timesteps: mx.array,
        dim: int,
        max_period: int = 10000,
    ) -> mx.array:
        """Create sinusoidal timestep embeddings.

        Args:
            timesteps: Timestep values [batch]
            dim: Embedding dimension
            max_period: Maximum period for frequency

        Returns:
            Embeddings [batch, dim]
        """
        half_dim = dim // 2
        freqs = mx.exp(
            -math.log(max_period) * mx.arange(half_dim, dtype=mx.float32) / half_dim
        )

        args = timesteps[:, None].astype(mx.float32) * freqs[None, :]
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)

        if dim % 2:
            embedding = mx.concatenate([embedding, mx.zeros((timesteps.shape[0], 1))], axis=-1)

        return embedding


class HunyuanPatchEmbed(nn.Module):
    """Patch embedding for converting latent images to sequences.

    Converts [batch, channels, height, width] to [batch, num_patches, hidden_dim].
    Uses a simple convolution with stride = patch_size.
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_dim: int = 1408,
        patch_size: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size

        # Patch embedding as a convolution
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def __call__(self, latents: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            latents: VAE latents [batch, channels, height, width]

        Returns:
            Patch embeddings [batch, num_patches, hidden_dim]
        """
        # Apply patch embedding convolution
        # MLX Conv2d expects [batch, H, W, channels] format
        latents = latents.transpose(0, 2, 3, 1)  # [batch, H, W, C]
        hidden_states = self.proj(latents)  # [batch, H//patch, W//patch, hidden_dim]

        # Flatten spatial dimensions to sequence
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(batch_size, -1, self.hidden_dim)

        return hidden_states


class HunyuanTextProjector(nn.Module):
    """Projects T5 text encoder output to cross-attention dimension.

    The HuggingFace Hunyuan model uses text_embedder which is a 2-layer MLP:
    T5 output (2048) -> linear_1 (8192) -> GELU -> linear_2 (1024)
    """

    def __init__(
        self,
        t5_dim: int = 2048,
        intermediate_dim: int = 8192,
        out_dim: int = 1024,
    ):
        super().__init__()
        self.linear_1 = nn.Linear(t5_dim, intermediate_dim, bias=True)
        self.linear_2 = nn.Linear(intermediate_dim, out_dim, bias=True)

    def __call__(self, t5_embeds: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            t5_embeds: T5 text embeddings [batch, seq_len, 2048]

        Returns:
            Projected text embeddings [batch, seq_len, 1024]
        """
        hidden = self.linear_1(t5_embeds)
        hidden = nn.gelu(hidden)
        hidden = self.linear_2(hidden)
        return hidden


class HunyuanOutputAdaLN(nn.Module):
    """AdaLN for output layer (norm_out).

    This is a simple linear projection that produces scale and shift for the
    final layer norm. The actual norm is applied separately.
    """

    def __init__(self, hidden_dim: int = 1408):
        super().__init__()
        # Linear to produce (shift, scale) from timestep embedding
        self.linear = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        self.hidden_dim = hidden_dim

    def __call__(self, hidden_states: mx.array, temb: mx.array) -> mx.array:
        """
        Apply adaptive layer norm to hidden states.

        Args:
            hidden_states: Input [batch, seq_len, hidden_dim]
            temb: Timestep embedding [batch, hidden_dim]

        Returns:
            Normalized output [batch, seq_len, hidden_dim]
        """
        # Get shift and scale from temb
        emb = self.linear(temb)  # [batch, 2 * hidden_dim]
        shift, scale = mx.split(emb, 2, axis=-1)

        # Apply layer norm manually
        mean = mx.mean(hidden_states, axis=-1, keepdims=True)
        var = mx.var(hidden_states, axis=-1, keepdims=True)
        normed = (hidden_states - mean) / mx.sqrt(var + 1e-6)

        # Apply shift and scale
        return normed * (1 + scale[:, None, :]) + shift[:, None, :]


class HunyuanDiT(nn.Module):
    """Main Hunyuan-DiT transformer model.

    Args:
        model_config: Model configuration
        num_blocks: Number of DiT transformer blocks (28 for Hunyuan)
        hidden_dim: Hidden dimension (1408)
        num_heads: Number of attention heads (16)
        head_dim: Attention head dimension (88)
        in_channels: Input latent channels (4)
        patch_size: Patch size for embedding (2)
        t5_dim: T5 encoder dimension (2048)
        text_proj_dim: Projected text dimension (1024)
    """

    def __init__(
        self,
        model_config: ModelConfig,
        num_blocks: int = 28,
        hidden_dim: int = 1408,
        num_heads: int = 16,
        head_dim: int = 88,
        in_channels: int = 4,
        patch_size: int = 2,
        t5_dim: int = 2048,
        text_proj_dim: int = 1024,
    ):
        super().__init__()
        self.model_config = model_config
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.patch_size = patch_size
        self.text_proj_dim = text_proj_dim

        # Timestep embedding
        self.time_embed = HunyuanTimestepEmbed(hidden_dim=hidden_dim)

        # Patch embedding for latent images
        self.patch_embed = HunyuanPatchEmbed(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            patch_size=patch_size,
        )

        # Text encoder projection (T5 -> cross-attention dim)
        self.text_proj = HunyuanTextProjector(
            t5_dim=t5_dim,
            intermediate_dim=8192,
            out_dim=text_proj_dim,
        )

        # DiT transformer blocks
        self.blocks = [
            HunyuanDiTBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                text_dim=text_proj_dim,  # Cross-attention uses projected text dim
            )
            for _ in range(num_blocks)
        ]

        # Output layers - norm_out is AdaLN
        self.norm_out = HunyuanOutputAdaLN(hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, patch_size * patch_size * in_channels, bias=True)

    def __call__(
        self,
        t: int,
        config: Config,
        hidden_states: mx.array,
        clip_embeds: mx.array,
        t5_embeds: mx.array,
        clip_attention_mask: mx.array | None = None,
        t5_attention_mask: mx.array | None = None,
    ) -> mx.array:
        """
        Forward pass of Hunyuan-DiT.

        Args:
            t: Current timestep index
            config: Configuration with scheduler and model info
            hidden_states: Noisy latents [batch, 4, H/8, W/8]
            clip_embeds: CLIP text embeddings (unused - Hunyuan uses T5 only)
            t5_embeds: T5 text embeddings [batch, 256, 2048]
            clip_attention_mask: Optional attention mask for CLIP (unused)
            t5_attention_mask: Optional attention mask for T5 [batch, 256]

        Returns:
            Predicted noise [batch, 4, H/8, W/8]
        """
        batch_size = hidden_states.shape[0]
        height = hidden_states.shape[2]
        width = hidden_states.shape[3]

        # 1. Get timestep embedding
        timestep = config.scheduler.timesteps[t].astype(config.precision)
        temb = self.time_embed(timestep)

        # 2. Patch embed the latent image
        hidden_states = self.patch_embed(hidden_states)

        # 3. Project T5 text embeddings
        encoder_hidden_states = self.text_proj(t5_embeds)

        # 4. Use T5 attention mask if provided
        attention_mask = t5_attention_mask

        # 5. Compute rotary embeddings for self-attention
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        rotary_emb = self._compute_rotary_embeddings(num_patches_h, num_patches_w)

        # 6. Run through DiT blocks
        for block in self.blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                rotary_emb=rotary_emb,
                attention_mask=attention_mask,
            )

        # 7. Final normalization (AdaLN) and projection
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # 8. Unpatchify back to image format
        hidden_states = self._unpatchify(hidden_states, height, width)

        return hidden_states

    def _compute_rotary_embeddings(
        self,
        num_patches_h: int,
        num_patches_w: int,
    ) -> mx.array:
        """Compute 2D rotary position embeddings for image patches.

        Args:
            num_patches_h: Number of patches in height
            num_patches_w: Number of patches in width

        Returns:
            Rotary embeddings [num_patches, head_dim//2, 2]
        """
        # Create position indices
        positions_h = mx.arange(num_patches_h)
        positions_w = mx.arange(num_patches_w)

        # Create 2D grid
        grid_h, grid_w = mx.meshgrid(positions_h, positions_w, indexing="ij")
        positions = mx.stack([grid_h.flatten(), grid_w.flatten()], axis=-1)  # [num_patches, 2]

        # Compute frequencies
        dim = self.head_dim // 2  # Half for H, half for W
        freqs = 1.0 / (10000 ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))

        # Compute sin/cos for each dimension
        pos_h = positions[:, 0:1]  # [num_patches, 1]
        pos_w = positions[:, 1:2]  # [num_patches, 1]

        # Embedding for height positions
        emb_h = pos_h.astype(mx.float32) * freqs[None, : dim // 2]  # [num_patches, dim//4]
        # Embedding for width positions
        emb_w = pos_w.astype(mx.float32) * freqs[None, : dim // 2]  # [num_patches, dim//4]

        # Concatenate H and W embeddings
        emb = mx.concatenate([emb_h, emb_w], axis=-1)  # [num_patches, dim//2]

        # Stack cos and sin
        cos_emb = mx.cos(emb)
        sin_emb = mx.sin(emb)
        rotary_emb = mx.stack([cos_emb, sin_emb], axis=-1)  # [num_patches, dim//2, 2]

        return rotary_emb

    def _unpatchify(
        self,
        hidden_states: mx.array,
        height: int,
        width: int,
    ) -> mx.array:
        """Convert patch sequence back to image format.

        Args:
            hidden_states: Patch sequence [batch, num_patches, patch_size**2 * channels]
            height: Original latent height
            width: Original latent width

        Returns:
            Image tensor [batch, channels, height, width]
        """
        batch_size = hidden_states.shape[0]
        channels = 4  # VAE latent channels
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size

        # Reshape to [batch, num_patches_h, num_patches_w, patch_size, patch_size, channels]
        hidden_states = hidden_states.reshape(
            batch_size, num_patches_h, num_patches_w,
            self.patch_size, self.patch_size, channels
        )

        # Rearrange to [batch, num_patches_h, patch_size, num_patches_w, patch_size, channels]
        hidden_states = hidden_states.transpose(0, 1, 3, 2, 4, 5)

        # Reshape to [batch, height, width, channels]
        hidden_states = hidden_states.reshape(batch_size, height, width, channels)

        # Transpose to [batch, channels, height, width]
        hidden_states = hidden_states.transpose(0, 3, 1, 2)

        return hidden_states
