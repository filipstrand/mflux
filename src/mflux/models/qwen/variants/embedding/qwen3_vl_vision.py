"""Vision Transformer for Qwen3-VL-2B Embedding and Reranker models.

This is specifically designed for Qwen3-VL-2B architecture which differs
from Qwen2-VL-7B in several ways:
- 24 vision blocks (not 32)
- Standard 2-layer MLP (not SwiGLU)
- LayerNorm with bias (not RMSNorm)
- Different merger structure
"""

import mlx.core as mx
from mlx import nn


class Qwen3VL2BVisionPatchEmbed(nn.Module):
    """Patch embedding for Qwen3-VL-2B.

    Uses 3D convolution to extract patches with temporal dimension.
    Includes bias (unlike some other implementations).
    """

    def __init__(
        self,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1024,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=[temporal_patch_size, patch_size, patch_size],
            stride=[temporal_patch_size, patch_size, patch_size],
            bias=True,  # Qwen3-VL has bias
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(
            batch_size, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        # MLX Conv3d expects: [N, D, H, W, C]
        hidden_states = hidden_states.transpose(0, 2, 3, 4, 1)
        output = self.proj(hidden_states)
        return output.reshape(batch_size, self.embed_dim)


class Qwen3VL2BVisionMLP(nn.Module):
    """Standard 2-layer MLP for Qwen3-VL-2B vision.

    Unlike SwiGLU, this uses a simple fc1 -> GELU -> fc2 structure.
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.linear_fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.linear_fc2 = nn.Linear(hidden_dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear_fc1(x)
        x = nn.gelu(x)
        x = self.linear_fc2(x)
        return x


class Qwen3VL2BVisionAttention(nn.Module):
    """Multi-head self-attention for Qwen3-VL-2B vision encoder."""

    def __init__(self, embed_dim: int = 1024, num_heads: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def _rotate_half(self, x: mx.array) -> mx.array:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return mx.concatenate([-x2, x1], axis=-1)

    def _apply_rope(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        cos_expanded = mx.expand_dims(cos, axis=0)
        sin_expanded = mx.expand_dims(sin, axis=0)
        rotated = (x * cos_expanded) + (self._rotate_half(x) * sin_expanded)
        return rotated

    def __call__(self, x: mx.array, position_embeddings=None, cu_seqlens=None) -> mx.array:
        seq_len, embed_dim = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = mx.split(qkv, 3, axis=1)
        q = q.squeeze(1).transpose(1, 0, 2)  # [heads, seq, head_dim]
        k = k.squeeze(1).transpose(1, 0, 2)
        v = v.squeeze(1).transpose(1, 0, 2)

        if position_embeddings is not None:
            cos_emb, sin_emb = position_embeddings
            if cos_emb.shape[0] != seq_len:
                cos_emb = cos_emb[:seq_len]
                sin_emb = sin_emb[:seq_len]
            q = self._apply_rope(q, cos_emb, sin_emb)
            k = self._apply_rope(k, cos_emb, sin_emb)

        # Scaled dot-product attention
        scale = 1.0 / (self.head_dim**0.5)
        q_batch = q[None, :, :, :]
        k_batch = k[None, :, :, :]
        v_batch = v[None, :, :, :]
        attn_output = mx.fast.scaled_dot_product_attention(q_batch, k_batch, v_batch, scale=scale)
        attn_output = attn_output[0]  # Remove batch dim

        # Reshape and project
        attn_output = attn_output.transpose(1, 0, 2).reshape(seq_len, embed_dim)
        return self.proj(attn_output)


class Qwen3VL2BVisionBlock(nn.Module):
    """Vision transformer block for Qwen3-VL-2B.

    Uses LayerNorm (with bias) instead of RMSNorm.
    """

    def __init__(self, embed_dim: int = 1024, num_heads: int = 16, mlp_ratio: float = 4.0):
        super().__init__()
        # LayerNorm with bias (Qwen3-VL uses this, not RMSNorm)
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = Qwen3VL2BVisionAttention(embed_dim, num_heads)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = Qwen3VL2BVisionMLP(embed_dim, mlp_hidden_dim)

    def __call__(self, x: mx.array, position_embeddings=None, cu_seqlens=None) -> mx.array:
        normed1 = self.norm1(x)
        attn_out = self.attn(normed1, position_embeddings, cu_seqlens)
        x = x + attn_out
        normed2 = self.norm2(x)
        mlp_out = self.mlp(normed2)
        x = x + mlp_out
        return x


class Qwen3VL2BMerger(nn.Module):
    """Patch merger for Qwen3-VL-2B.

    Merges spatial patches and projects to language model hidden size.

    Architecture (from actual weights):
    1. Apply LayerNorm to patches (1024 dim) BEFORE merging
    2. Spatial merge: reshape groups of 4 patches into single vectors (4096 dim)
    3. MLP: linear_fc1 (4096 -> 4096) -> GELU -> linear_fc2 (4096 -> 2048)
    """

    def __init__(self, context_dim: int = 1024, hidden_size: int = 2048, spatial_merge_size: int = 2):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.hidden_size_merged = context_dim * (spatial_merge_size**2)  # 4096

        # LayerNorm BEFORE spatial merging (on context_dim, not merged)
        self.norm = nn.LayerNorm(context_dim, eps=1e-6)
        # MLP: merged_dim -> merged_dim -> hidden_size
        self.linear_fc1 = nn.Linear(self.hidden_size_merged, self.hidden_size_merged, bias=True)
        self.linear_fc2 = nn.Linear(self.hidden_size_merged, hidden_size, bias=True)

    def __call__(self, x: mx.array, grid_thw: mx.array) -> mx.array:
        # Apply norm BEFORE spatial merging
        x = self.norm(x)

        # Spatial merge: group patches into larger vectors
        merged_patches = []
        offset = 0
        for t, h, w in grid_thw:
            t, h, w = int(t), int(h), int(w)
            num_patches = t * h * w
            x_img = x[offset : offset + num_patches]
            x_merged = x_img.reshape(-1, self.hidden_size_merged)
            merged_patches.append(x_merged)
            offset += num_patches

        x = mx.concatenate(merged_patches, axis=0)
        x = self.linear_fc1(x)
        x = nn.gelu(x)
        x = self.linear_fc2(x)
        return x


class Qwen3VL2BVisionRotaryEmbedding(nn.Module):
    """Rotary position embeddings for vision transformer."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        self.inv_freq = inv_freq

    def __call__(self, max_grid_size: int) -> mx.array:
        positions = mx.arange(max_grid_size, dtype=mx.float32)
        freqs = mx.outer(positions, self.inv_freq)
        return freqs


class Qwen3VL2BVisionTransformer(nn.Module):
    """Complete vision transformer for Qwen3-VL-2B.

    Architecture:
    - 24 transformer blocks (not 32)
    - embed_dim = 1024
    - num_heads = 16
    - mlp_ratio = 4.0 (hidden = 4096)
    - LayerNorm (not RMSNorm)
    - Standard MLP (not SwiGLU)
    """

    def __init__(
        self,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        hidden_size: int = 2048,
        spatial_merge_size: int = 2,
    ):
        super().__init__()

        self.patch_embed = Qwen3VL2BVisionPatchEmbed(patch_size, temporal_patch_size, in_channels, embed_dim)
        self.spatial_merge_size = spatial_merge_size
        self.spatial_merge_unit = spatial_merge_size * spatial_merge_size
        self.patch_size = patch_size

        head_dim = embed_dim // num_heads
        self.rotary_pos_emb = Qwen3VL2BVisionRotaryEmbedding(head_dim // 2)

        self.blocks = [Qwen3VL2BVisionBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        self.merger = Qwen3VL2BMerger(embed_dim, hidden_size, spatial_merge_size)

    def rot_pos_emb(self, grid_thw: mx.array) -> mx.array:
        pos_ids = []
        for t, h, w in grid_thw:
            t, h, w = int(t), int(h), int(w)
            hpos_ids = mx.repeat(mx.arange(h)[..., None], w, axis=1)
            wpos_ids = mx.repeat(mx.arange(w)[None, ...], h, axis=0)
            merge_h = h // self.spatial_merge_size
            merge_w = w // self.spatial_merge_size
            hpos_ids = hpos_ids.reshape(merge_h, self.spatial_merge_size, merge_w, self.spatial_merge_size)
            wpos_ids = wpos_ids.reshape(merge_h, self.spatial_merge_size, merge_w, self.spatial_merge_size)
            hpos_ids = mx.transpose(hpos_ids, (0, 2, 1, 3))
            wpos_ids = mx.transpose(wpos_ids, (0, 2, 1, 3))
            hpos_ids = hpos_ids.reshape(-1)
            wpos_ids = wpos_ids.reshape(-1)
            pos_id_pair = mx.stack([hpos_ids, wpos_ids], axis=-1)
            pos_id_pair = mx.tile(pos_id_pair, (t, 1))
            pos_ids.append(pos_id_pair)
        pos_ids = mx.concatenate(pos_ids, axis=0)
        max_grid_size = int(mx.max(grid_thw[:, 1:]).item())
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        h_indices = pos_ids[:, 0].astype(mx.int32)
        w_indices = pos_ids[:, 1].astype(mx.int32)
        h_emb = rotary_pos_emb_full[h_indices]
        w_emb = rotary_pos_emb_full[w_indices]
        rotary_pos_emb = mx.stack([h_emb, w_emb], axis=1)
        rotary_pos_emb = rotary_pos_emb.reshape(rotary_pos_emb.shape[0], -1)
        emb = mx.concatenate([rotary_pos_emb, rotary_pos_emb], axis=-1)
        position_embeddings = (mx.cos(emb), mx.sin(emb))
        return position_embeddings

    def __call__(self, pixel_values: mx.array, grid_thw: mx.array) -> mx.array:
        hidden_states = self.patch_embed(pixel_values)
        position_embeddings = self.rot_pos_emb(grid_thw)

        # Forward through blocks
        for block in self.blocks:
            hidden_states = block(hidden_states, position_embeddings)

        # Merge patches and project to language model hidden size
        hidden_states = self.merger(hidden_states, grid_thw)
        return hidden_states
