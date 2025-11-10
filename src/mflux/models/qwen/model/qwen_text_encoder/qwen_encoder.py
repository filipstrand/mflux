import mlx.core as mx
from mlx import nn

from mflux.models.qwen.model.qwen_text_encoder.qwen_encoder_layer import QwenEncoderLayer
from mflux.models.qwen.model.qwen_text_encoder.qwen_rms_norm import QwenRMSNorm
from mflux.models.qwen.model.qwen_text_encoder.qwen_rope import QwenRotaryEmbedding


class VisionPatchEmbed(nn.Module):
    """Vision patch embedding using 3D convolution - matches visual.patch_embed.proj"""

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1280,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # Conv3d: [embed_dim, in_channels, temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=[temporal_patch_size, patch_size, patch_size],
            stride=[temporal_patch_size, patch_size, patch_size],
            bias=False,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        # IMPORTANT: Match PyTorch's approach exactly to avoid subtle bugs
        # PyTorch receives flattened [N, C*T*H*W], reshapes to [N, C, T, H, W], then Conv3d
        # We do the same: keep PyTorch's NCDHW format, only transpose for Conv3d, then flatten back

        batch_size = hidden_states.shape[0]

        # Input is flattened: [num_patches, in_channels * temporal_patch_size * patch_size * patch_size]
        # Reshape to PyTorch format (NCDHW): [num_patches, in_channels, temporal, height, width]
        hidden_states = hidden_states.reshape(
            batch_size, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )

        # Transpose to MLX Conv3d format (NDHWC) only for the convolution
        # [num_patches, in_channels, temporal, height, width] -> [num_patches, temporal, height, width, in_channels]
        hidden_states = hidden_states.transpose(0, 2, 3, 4, 1)

        # Apply Conv3d (weights are already in MLX format from weight loading)
        output = self.proj(hidden_states)  # [num_patches, 1, 1, 1, embed_dim]

        # Flatten to match PyTorch output: [num_patches, embed_dim]
        return output.reshape(batch_size, self.embed_dim)


class VisionRotaryEmbedding(nn.Module):
    """Vision rotary positional embedding"""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        # Create inverse frequencies
        inv_freq = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        self.inv_freq = inv_freq

    def __call__(self, max_grid_size: int) -> mx.array:
        # Create position embeddings for max_grid_size
        # Returns only freqs (NOT cos/sin concatenated) - matching HF
        positions = mx.arange(max_grid_size, dtype=mx.float32)
        freqs = mx.outer(positions, self.inv_freq)
        return freqs  # [max_grid_size, dim//2]


class VisionMLP(nn.Module):
    """
    Vision MLP block - GLU-style MLP matching Qwen2_5_VLMLP

    Uses gated linear unit architecture:
    output = down_proj(silu(gate_proj(x)) * up_proj(x))

    This is NOT a simple 2-layer MLP! It's a 3-layer gated architecture.
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        # GLU-style MLP with 3 projections (SwiGLU variant)
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=True)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=True)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        # GLU formula: down_proj(silu(gate_proj(x)) * up_proj(x))
        # silu(x) = x * sigmoid(x), also known as Swish
        gate = nn.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class VisionAttention(nn.Module):
    """Vision attention block - matches visual.blocks.*.attn"""

    def __init__(self, embed_dim: int = 1280, num_heads: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def __call__(self, x: mx.array, position_embeddings=None, cu_seqlens=None) -> mx.array:
        # For vision attention, x is [seq_len, embed_dim] (no batch dimension)
        seq_len, embed_dim = x.shape

        # Compute QKV
        qkv = self.qkv(x)  # [seq_len, 3 * embed_dim]
        qkv = qkv.reshape(seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = mx.split(qkv, 3, axis=1)
        q = q.squeeze(1).transpose(1, 0, 2)  # [heads, seq, head_dim]
        k = k.squeeze(1).transpose(1, 0, 2)
        v = v.squeeze(1).transpose(1, 0, 2)

        # Apply rotary position embeddings if provided
        if position_embeddings is not None:
            cos_emb, sin_emb = position_embeddings
            # Ensure embeddings match sequence length
            if cos_emb.shape[0] != seq_len:
                cos_emb = cos_emb[:seq_len]
                sin_emb = sin_emb[:seq_len]

            # Debug checkpoint: Verify position embeddings before RoPE
            from mflux_debugger.semantic_checkpoint import debug_checkpoint

            debug_checkpoint(
                "mlx_vision_attention_before_rope",
                skip=True,
                q_shape=q.shape,
                k_shape=k.shape,
                q_preview=q[0, :3, :5].tolist() if q.shape[0] > 0 and q.shape[1] > 0 else None,
                k_preview=k[0, :3, :5].tolist() if k.shape[0] > 0 and k.shape[1] > 0 else None,
                cos_shape=cos_emb.shape,
                sin_shape=sin_emb.shape,
                cos_preview=cos_emb[:3, :5].tolist() if cos_emb.shape[0] > 0 else None,
                sin_preview=sin_emb[:3, :5].tolist() if sin_emb.shape[0] > 0 else None,
            )

            # Apply RoPE to q and k (matching HF's rotate_half logic)
            # Matching QwenAttention._apply_multimodal_rotary_pos_emb implementation
            def apply_rope(x, cos, sin):
                # x: [heads, seq, head_dim], cos/sin: [seq, head_dim]
                # Expand cos/sin to match head dimension for broadcasting
                cos_expanded = mx.expand_dims(cos, axis=0)  # [1, seq, head_dim]
                sin_expanded = mx.expand_dims(sin, axis=0)  # [1, seq, head_dim]

                # Standard RoPE formula: (q * cos) + (rotate_half(q) * sin)
                # where rotate_half([x1, x2]) = [-x2, x1]
                def rotate_half(x):
                    x1 = x[..., : x.shape[-1] // 2]
                    x2 = x[..., x.shape[-1] // 2 :]
                    return mx.concatenate([-x2, x1], axis=-1)

                rotated = (x * cos_expanded) + (rotate_half(x) * sin_expanded)
                return rotated

            q = apply_rope(q, cos_emb, sin_emb)
            k = apply_rope(k, cos_emb, sin_emb)

            # Debug checkpoint: Verify q and k after RoPE
            debug_checkpoint(
                "mlx_vision_attention_after_rope",
                skip=True,
                q_shape=q.shape,
                k_shape=k.shape,
                q_preview=q[0, :3, :5].tolist() if q.shape[0] > 0 and q.shape[1] > 0 else None,
                k_preview=k[0, :3, :5].tolist() if k.shape[0] > 0 and k.shape[1] > 0 else None,
            )

        # Process attention chunks if cu_seqlens is provided (windowed attention)
        if cu_seqlens is not None and len(cu_seqlens) > 2:
            # Split Q, K, V into chunks based on cu_seqlens
            # cu_seqlens is cumulative, so lengths[i] = cu_seqlens[i+1] - cu_seqlens[i]
            lengths = [int((cu_seqlens[i + 1] - cu_seqlens[i]).item()) for i in range(len(cu_seqlens) - 1)]

            # Split tensors (q,k,v are [heads, seq, head_dim])
            q_chunks = []
            k_chunks = []
            v_chunks = []
            offset = 0
            for length in lengths:
                q_chunks.append(q[:, offset : offset + length, :])
                k_chunks.append(k[:, offset : offset + length, :])
                v_chunks.append(v[:, offset : offset + length, :])
                offset += length

            # Process each chunk separately
            attn_outputs = []
            scale = 1.0 / (self.head_dim**0.5)
            for q_chunk, k_chunk, v_chunk in zip(q_chunks, k_chunks, v_chunks):
                # Compute attention for this chunk
                scores = mx.matmul(q_chunk, k_chunk.transpose(0, 2, 1)) * scale
                attn_weights = mx.softmax(scores, axis=-1)
                attn_chunk = mx.matmul(attn_weights, v_chunk)  # [heads, chunk_len, head_dim]
                attn_outputs.append(attn_chunk)

            # Concatenate chunks back together
            attn_output = mx.concatenate(attn_outputs, axis=1)  # [heads, seq, head_dim]
        else:
            # Full attention (no chunking)
            scale = 1.0 / (self.head_dim**0.5)
            scores = mx.matmul(q, k.transpose(0, 2, 1)) * scale  # [heads, seq, seq]
            attn_weights = mx.softmax(scores, axis=-1)
            attn_output = mx.matmul(attn_weights, v)  # [heads, seq, head_dim]

        # Reshape and project
        attn_output = attn_output.transpose(1, 0, 2).reshape(seq_len, embed_dim)  # [seq, embed_dim]
        return self.proj(attn_output)


class VisionBlock(nn.Module):
    """Vision transformer block - matches visual.blocks.*"""

    def __init__(self, embed_dim: int = 1280, num_heads: int = 16, mlp_ratio: float = 2.671875):
        super().__init__()
        self.norm1 = nn.RMSNorm(embed_dim, eps=1e-6)  # Fixed: was LayerNorm, should be RMSNorm
        self.norm2 = nn.RMSNorm(embed_dim, eps=1e-6)  # Fixed: was LayerNorm, should be RMSNorm
        self.attn = VisionAttention(embed_dim, num_heads)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)  # 3420 for 1280 * 2.671875
        self.mlp = VisionMLP(embed_dim, mlp_hidden_dim)

    def __call__(self, x: mx.array, position_embeddings=None, cu_seqlens=None) -> mx.array:
        # Pre-norm architecture
        # TEMP DEBUG: Track divergence within block (expanded for breakpoint tracing)
        normed1 = self.norm1(x)
        attn_out = self.attn(normed1, position_embeddings, cu_seqlens)
        x = x + attn_out

        normed2 = self.norm2(x)
        mlp_out = self.mlp(normed2)
        x = x + mlp_out
        return x


class PatchMerger(nn.Module):
    """
    Spatial patch merger - matches HF's Qwen2_5_VLPatchMerger.
    Merges spatial_merge_size x spatial_merge_size patches into one patch.
    """

    def __init__(self, context_dim: int, hidden_size: int, spatial_merge_size: int = 2):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.hidden_size_merged = context_dim * (spatial_merge_size**2)

        # RMSNorm for query (HF uses Qwen2RMSNorm, confirmed via runtime debugging)
        self.ln_q = nn.RMSNorm(context_dim, eps=1e-6)

        # MLP: merged_dim -> merged_dim -> output_dim
        self.mlp_0 = nn.Linear(self.hidden_size_merged, self.hidden_size_merged, bias=True)
        self.mlp_1 = nn.Linear(self.hidden_size_merged, hidden_size, bias=True)

    def __call__(self, x: mx.array, grid_thw: mx.array) -> mx.array:
        """
        Args:
            x: [num_patches, context_dim]
            grid_thw: [num_images, 3] - temporal, height, width dimensions
        Returns:
            [num_patches // (spatial_merge_size**2), hidden_size]
        """
        # Apply RMSNorm
        # Debug: Log weight values for verification
        if not hasattr(self, "_weights_logged"):
            from mflux_debugger.semantic_checkpoint import debug_checkpoint

            debug_checkpoint(
                "mlx_patch_merger_weights",
                skip=True,
                ln_q_weight_preview=self.ln_q.weight[:10].tolist(),
                mlp_0_weight_preview=self.mlp_0.weight[0, :10].tolist(),
                mlp_0_bias_preview=self.mlp_0.bias[:10].tolist(),
                mlp_1_weight_preview=self.mlp_1.weight[0, :10].tolist(),
                mlp_1_bias_preview=self.mlp_1.bias[:10].tolist(),
            )
            self._weights_logged = True
        x = self.ln_q(x)

        # CRITICAL: Match PyTorch's simple approach - just reshape consecutive patches
        # PyTorch does: x.view(-1, self.hidden_size) which merges consecutive groups of 4 patches
        # After window reordering, patches are already grouped into 2x2 spatial blocks,
        # so merging consecutive groups correctly merges spatially adjacent patches
        # Process each image in grid_thw separately (matching the original logic)
        merged_patches = []
        offset = 0
        for t, h, w in grid_thw:
            t, h, w = int(t), int(h), int(w)
            num_patches_this_image = t * h * w
            x_this_image = x[offset : offset + num_patches_this_image]  # [t*h*w, context_dim]

            # Simple reshape to merge consecutive groups of 4 patches (matching PyTorch)
            # [t*h*w, context_dim] -> [t*h*w//4, context_dim*4] = [num_merged_patches, hidden_size_merged]
            x_merged = x_this_image.reshape(-1, self.hidden_size_merged)
            merged_patches.append(x_merged)
            offset += num_patches_this_image

        # Concatenate all images
        x = mx.concatenate(merged_patches, axis=0)  # [total_merged_patches, hidden_size_merged]

        # Debug checkpoint after concatenation
        from mflux_debugger.semantic_checkpoint import debug_checkpoint

        debug_checkpoint(
            "mlx_patch_merger_after_concat",
            skip=True,
            x_shape=x.shape,
            x_preview=x[:5, :10].tolist() if x.shape[0] > 0 else None,
        )

        # Apply MLP with GELU activation
        x = self.mlp_0(x)
        debug_checkpoint(
            "mlx_patch_merger_after_mlp_0",
            skip=True,
            x_shape=x.shape,
            x_preview=x[:5, :10].tolist() if x.shape[0] > 0 else None,
        )
        x = nn.gelu(x)
        debug_checkpoint(
            "mlx_patch_merger_after_gelu",
            skip=True,
            x_shape=x.shape,
            x_preview=x[:5, :10].tolist() if x.shape[0] > 0 else None,
        )
        x = self.mlp_1(x)
        debug_checkpoint(
            "mlx_patch_merger_after_mlp_1",
            skip=True,
            x_shape=x.shape,
            x_preview=x[:5, :10].tolist() if x.shape[0] > 0 else None,
        )

        return x


class VisionTransformer(nn.Module):
    """Complete vision transformer - matches the visual.* structure"""

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1280,
        depth: int = 32,
        num_heads: int = 16,
        mlp_ratio: float = 2.671875,
        hidden_size: int = 3584,
        spatial_merge_size: int = 2,
        window_size: int = 112,
        fullatt_block_indexes: list = None,
    ):
        super().__init__()

        self.patch_embed = VisionPatchEmbed(patch_size, temporal_patch_size, in_channels, embed_dim)
        self.spatial_merge_size = spatial_merge_size
        self.window_size = window_size
        self.fullatt_block_indexes = fullatt_block_indexes or [7, 15, 23, 31]
        self.spatial_merge_unit = spatial_merge_size * spatial_merge_size
        self.patch_size = patch_size

        head_dim = embed_dim // num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = [VisionBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        # Register blocks as attributes for parameter tracking
        for i, block in enumerate(self.blocks):
            setattr(self, f"blocks_{i}", block)

        # Use proper patch merger to match HF
        self.merger = PatchMerger(embed_dim, hidden_size, spatial_merge_size)

    def get_window_index(self, grid_thw: mx.array):
        """
        Compute window indices and cumulative sequence lengths for windowed attention.
        Matches HF's get_window_index method.

        Returns:
            window_index: Indices for spatial windowing (for reordering patches)
            cu_window_seqlens: Cumulative sequence lengths for windowed attention chunks
        """
        window_index = []
        cu_window_seqlens = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.patch_size // self.spatial_merge_size

        for t, grid_h, grid_w in grid_thw:
            t, grid_h, grid_w = int(t), int(grid_h), int(grid_w)
            llm_grid_h = grid_h // self.spatial_merge_size
            llm_grid_w = grid_w // self.spatial_merge_size

            # Create index array for this image
            index = mx.arange(t * llm_grid_h * llm_grid_w).reshape(t, llm_grid_h, llm_grid_w)

            # Compute padding for windowing
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size

            # Pad the index
            index_padded = mx.pad(index, ((0, 0), (0, pad_h), (0, pad_w)), constant_values=-100)

            # Reshape to create windows
            index_padded = index_padded.reshape(
                t, num_windows_h, vit_merger_window_size, num_windows_w, vit_merger_window_size
            )
            index_padded = mx.transpose(index_padded, (0, 1, 3, 2, 4)).reshape(
                t, num_windows_h * num_windows_w, vit_merger_window_size, vit_merger_window_size
            )

            # Compute sequence lengths per window
            seqlens = mx.sum((index_padded != -100).astype(mx.int32), axis=(2, 3)).reshape(-1)
            index_padded_flat = index_padded.reshape(-1)

            # MLX doesn't support boolean indexing, convert to numpy, filter, convert back
            import numpy as np

            index_padded_np = np.array(index_padded_flat)
            index_new = mx.array(index_padded_np[index_padded_np != -100])

            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = mx.cumsum(seqlens) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += t * llm_grid_h * llm_grid_w

        window_index = mx.concatenate(window_index, axis=0)
        cu_window_seqlens = mx.array(cu_window_seqlens, dtype=mx.int32)

        return window_index, cu_window_seqlens

    def rot_pos_emb(self, grid_thw: mx.array) -> mx.array:
        """Generate rotary position embeddings with spatial merging (matching HF)"""
        pos_ids = []
        for t, h, w in grid_thw:
            t, h, w = int(t), int(h), int(w)

            # Create position IDs with spatial merging (matching HF's approach)
            # First create h and w position grids
            hpos_ids = mx.repeat(mx.arange(h)[..., None], w, axis=1)  # [h, w]
            wpos_ids = mx.repeat(mx.arange(w)[None, ...], h, axis=0)  # [h, w]

            # Reshape to group spatial_merge_size x spatial_merge_size patches
            merge_h = h // self.spatial_merge_size
            merge_w = w // self.spatial_merge_size

            # Reshape: [h, w] -> [merge_h, spatial_merge_size, merge_w, spatial_merge_size]
            hpos_ids = hpos_ids.reshape(merge_h, self.spatial_merge_size, merge_w, self.spatial_merge_size)
            wpos_ids = wpos_ids.reshape(merge_h, self.spatial_merge_size, merge_w, self.spatial_merge_size)

            # Permute: [merge_h, merge_w, spatial_merge_size, spatial_merge_size]
            hpos_ids = mx.transpose(hpos_ids, (0, 2, 1, 3))
            wpos_ids = mx.transpose(wpos_ids, (0, 2, 1, 3))

            # Flatten to [total_patches]
            hpos_ids = hpos_ids.reshape(-1)
            wpos_ids = wpos_ids.reshape(-1)

            # Stack and repeat for temporal dimension
            pos_id_pair = mx.stack([hpos_ids, wpos_ids], axis=-1)  # [h*w, 2]
            pos_id_pair = mx.tile(pos_id_pair, (t, 1))  # [t*h*w, 2]
            pos_ids.append(pos_id_pair)

        pos_ids = mx.concatenate(pos_ids, axis=0)  # [total_patches, 2]

        # Get max grid size for rotary embeddings
        max_grid_size = int(mx.max(grid_thw[:, 1:]).item())
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)  # [max_grid_size, dim//2]

        # Index into the full embedding using fancy indexing (matching HF exactly)
        # pos_ids is [N, 2] with (h, w) coordinates
        # rotary_pos_emb_full[pos_ids] gives [N, 2, dim//2]
        h_indices = pos_ids[:, 0].astype(mx.int32)
        w_indices = pos_ids[:, 1].astype(mx.int32)
        h_emb = rotary_pos_emb_full[h_indices]  # [total_patches, dim//2]
        w_emb = rotary_pos_emb_full[w_indices]  # [total_patches, dim//2]

        # Stack and flatten to get [N, dim] (h and w embeddings interleaved)
        rotary_pos_emb = mx.stack([h_emb, w_emb], axis=1)  # [N, 2, dim//2]
        rotary_pos_emb = rotary_pos_emb.reshape(rotary_pos_emb.shape[0], -1)  # [N, dim]

        return rotary_pos_emb

    def __call__(self, pixel_values: mx.array, grid_thw: mx.array) -> mx.array:
        """
        Args:
            pixel_values: [num_patches, channels * temporal_patch_size * patch_size * patch_size] (flattened)
            grid_thw: [num_images, 3] - temporal, height, width dimensions
        """
        from mflux_debugger.semantic_checkpoint import debug_checkpoint

        # Checkpoint before vision transformer to compare inputs
        debug_checkpoint(
            "mlx_before_vision_transformer",
            skip=True,
            pixel_values_shape=pixel_values.shape,
            pixel_values_preview=pixel_values[:10, :10].tolist() if pixel_values.shape[0] > 0 else None,
            pixel_values_mean=float(mx.mean(pixel_values).item()),
            pixel_values_std=float(mx.std(pixel_values).item()),
            grid_thw=grid_thw.tolist(),
        )

        # Patch embedding (will reshape internally to match PyTorch's NCDHW format)
        hidden_states = self.patch_embed(pixel_values)  # [num_patches, embed_dim]

        print(f"🔎 VisionTransformer: After patch_embed: {hidden_states.shape}")
        # Checkpoint after patch_embed (verified - patch_embed matches)
        debug_checkpoint(
            "mlx_after_patch_embed",
            skip=True,  # Verified - matches PyTorch
            verified=True,
            hidden_states_shape=hidden_states.shape,
            hidden_states_preview=hidden_states[:5, :10].tolist() if hidden_states.shape[0] > 0 else None,
        )

        # Rotary position embeddings (with spatial merging order)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)  # [num_patches, dim] where dim = head_dim

        debug_checkpoint(
            "mlx_after_rotary_emb",
            skip=True,
            rotary_pos_emb_shape=rotary_pos_emb.shape,
            rotary_pos_emb_preview=rotary_pos_emb[:5, :10].tolist() if rotary_pos_emb.shape[0] > 0 else None,
        )

        # Get window index and cu_seqlens for windowed attention
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)

        debug_checkpoint(
            "mlx_after_window_index",
            skip=True,
            window_index_shape=window_index.shape,
            window_index_preview=window_index[:20].tolist() if window_index.shape[0] > 0 else None,
            cu_window_seqlens_preview=cu_window_seqlens[:10].tolist() if cu_window_seqlens.shape[0] > 0 else None,
        )

        # Ensure cu_window_seqlens has unique consecutive values (matching HF's torch.unique_consecutive)
        cu_window_seqlens_unique = [cu_window_seqlens[0].item()]
        for i in range(1, len(cu_window_seqlens)):
            if cu_window_seqlens[i].item() != cu_window_seqlens_unique[-1]:
                cu_window_seqlens_unique.append(cu_window_seqlens[i].item())
        cu_window_seqlens = mx.array(cu_window_seqlens_unique, dtype=mx.int32)

        # Compute cu_seqlens for full attention
        seq_len = hidden_states.shape[0]
        cu_seqlens = []
        offset = 0
        for t, h, w in grid_thw:
            t, h, w = int(t), int(h), int(w)
            length = t * h * w
            offset += length
            cu_seqlens.append(offset)
        cu_seqlens = mx.array([0] + cu_seqlens, dtype=mx.int32)

        # Reorder hidden_states according to window_index (matching PyTorch logic)
        # CRITICAL: window_index is for GROUPS (after grouping), not individual patches
        # PyTorch groups consecutive patches: reshape(seq_len // spatial_merge_unit, spatial_merge_unit, -1)
        # This means groups are: [0,1,2,3], [4,5,6,7], [8,9,10,11], ... (consecutive patches in row-major order)
        # NOT 2x2 spatial blocks! The window_index expects this consecutive grouping.

        debug_checkpoint(
            "mlx_before_window_reorder",
            skip=False,  # Verify raw patches before grouping
            hidden_states_shape=hidden_states.shape,
            hidden_states_preview=hidden_states[:5, :10].tolist() if hidden_states.shape[0] > 0 else None,
            window_index_shape=window_index.shape,
            window_index_preview=window_index[:20].tolist() if window_index.shape[0] > 0 else None,
            spatial_merge_unit=self.spatial_merge_unit,
            seq_len=hidden_states.shape[0],
            num_groups=hidden_states.shape[0] // self.spatial_merge_unit,
        )

        # CRITICAL FIX: Match PyTorch's simple consecutive grouping
        # PyTorch does: reshape(seq_len // spatial_merge_unit, spatial_merge_unit, -1)
        # This groups consecutive patches, NOT 2x2 spatial blocks!
        # The window_index expects groups in this consecutive order
        seq_len = hidden_states.shape[0]
        num_groups = seq_len // self.spatial_merge_unit
        hidden_states_grouped = hidden_states.reshape(
            num_groups, self.spatial_merge_unit, -1
        )  # [total_groups, spatial_merge_unit, hidden_dim]

        debug_checkpoint(
            "mlx_after_grouping_before_reorder",
            skip=False,  # Verify groups are correct (consecutive patches)
            hidden_states_grouped_shape=hidden_states_grouped.shape,
            hidden_states_grouped_preview=hidden_states_grouped[:3, :, :5].tolist()
            if hidden_states_grouped.shape[0] > 0
            else None,
            first_group_indices=list(range(self.spatial_merge_unit)),  # Should be [0,1,2,3]
            second_group_indices=list(
                range(self.spatial_merge_unit, 2 * self.spatial_merge_unit)
            ),  # Should be [4,5,6,7]
        )

        # Now reorder groups using window_index
        hidden_states_grouped = hidden_states_grouped[window_index.astype(mx.int32), :, :]

        debug_checkpoint(
            "mlx_after_reorder_before_reshape",
            skip=False,  # Verify reordered groups
            hidden_states_grouped_shape=hidden_states_grouped.shape,
            hidden_states_grouped_preview=hidden_states_grouped[:3, :, :5].tolist()
            if hidden_states_grouped.shape[0] > 0
            else None,
            window_index_first_10=window_index[:10].tolist() if window_index.shape[0] > 0 else None,
        )

        # Reshape back to individual patches: [total_groups, spatial_merge_unit, hidden_dim] -> [total_patches, hidden_dim]
        hidden_states = hidden_states_grouped.reshape(seq_len, -1)

        # Checkpoint after window reorder (critical - check if reordering matches PyTorch)
        debug_checkpoint(
            "mlx_after_window_reorder",
            skip=False,  # Verify final reordered patches match PyTorch
            hidden_states_shape=hidden_states.shape,
            hidden_states_preview=hidden_states[:5, :10].tolist() if hidden_states.shape[0] > 0 else None,
        )

        # CRITICAL FIX: Match PyTorch's simple consecutive grouping for rotary embeddings too
        # Same as hidden_states - group consecutive patches
        rotary_pos_emb_grouped = rotary_pos_emb.reshape(
            num_groups, self.spatial_merge_unit, -1
        )  # [total_groups, spatial_merge_unit, hidden_dim]

        debug_checkpoint(
            "mlx_rotary_after_grouping_before_reorder",
            skip=False,  # Verify rotary groups match hidden_states grouping
            rotary_pos_emb_grouped_shape=rotary_pos_emb_grouped.shape,
            rotary_pos_emb_grouped_preview=rotary_pos_emb_grouped[:3, :, :5].tolist()
            if rotary_pos_emb_grouped.shape[0] > 0
            else None,
        )

        # Now reorder groups using window_index
        rotary_pos_emb_grouped = rotary_pos_emb_grouped[window_index.astype(mx.int32), :, :]

        # Reshape back to individual patches: [total_groups, spatial_merge_unit, hidden_dim] -> [total_patches, hidden_dim]
        rotary_pos_emb = rotary_pos_emb_grouped.reshape(seq_len, -1)

        # Duplicate embeddings (matching HF: torch.cat((emb, emb), dim=-1))
        emb = mx.concatenate([rotary_pos_emb, rotary_pos_emb], axis=-1)  # [num_patches, dim*2]
        position_embeddings = (mx.cos(emb), mx.sin(emb))

        debug_checkpoint(
            "mlx_after_position_embeddings",
            skip=True,
            position_embeddings_cos_shape=position_embeddings[0].shape,
            position_embeddings_cos_preview=position_embeddings[0][:5, :10].tolist()
            if position_embeddings[0].shape[0] > 0
            else None,
            position_embeddings_sin_shape=position_embeddings[1].shape,
            position_embeddings_sin_preview=position_embeddings[1][:5, :10].tolist()
            if position_embeddings[1].shape[0] > 0
            else None,
        )

        # Apply vision transformer blocks with windowed/full attention switching
        for layer_num, block in enumerate(self.blocks):
            # Choose cu_seqlens based on whether this is a full attention layer
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens

            # Checkpoint before block
            if layer_num == 0:
                debug_checkpoint(
                    "mlx_before_first_vision_block",
                    skip=True,
                    hidden_states_shape=hidden_states.shape,
                    hidden_states_preview=hidden_states[:5, :10].tolist() if hidden_states.shape[0] > 0 else None,
                )

            hidden_states = block(hidden_states, position_embeddings, cu_seqlens_now)

            # Debug checkpoints at key layers
            if layer_num == 0:
                # Checkpoint after first vision block (critical - check if first block matches PyTorch)
                debug_checkpoint(
                    "mlx_after_first_vision_block",
                    skip=True,  # Verified - matches PyTorch (within bfloat16 precision)
                    verified=True,
                    hidden_states_shape=hidden_states.shape,
                    hidden_states_preview=hidden_states[:5, :10].tolist() if hidden_states.shape[0] > 0 else None,
                )
            elif layer_num == 7:  # First full attention layer
                debug_checkpoint(
                    "mlx_after_vision_block_7",
                    skip=True,
                    hidden_states_shape=hidden_states.shape,
                    hidden_states_preview=hidden_states[:5, :10].tolist() if hidden_states.shape[0] > 0 else None,
                )
            elif layer_num == 15:
                debug_checkpoint(
                    "mlx_after_vision_block_15",
                    skip=True,
                    hidden_states_shape=hidden_states.shape,
                    hidden_states_preview=hidden_states[:5, :10].tolist() if hidden_states.shape[0] > 0 else None,
                )
            elif layer_num == 23:
                debug_checkpoint(
                    "mlx_after_vision_block_23",
                    skip=True,
                    hidden_states_shape=hidden_states.shape,
                    hidden_states_preview=hidden_states[:5, :10].tolist() if hidden_states.shape[0] > 0 else None,
                )
            elif layer_num == 31:  # Last block
                debug_checkpoint(
                    "mlx_after_vision_block_31",
                    skip=True,
                    hidden_states_shape=hidden_states.shape,
                    hidden_states_preview=hidden_states[:5, :10].tolist() if hidden_states.shape[0] > 0 else None,
                )

        print(f"🔎 VisionTransformer: After transformer blocks: {hidden_states.shape}")
        print(f"🔎 VisionTransformer: After transformer blocks std: {float(mx.std(hidden_states)):.4f}")
        debug_checkpoint(
            "mlx_after_all_vision_blocks",
            skip=True,
            hidden_states_shape=hidden_states.shape,
            hidden_states_preview=hidden_states[:5, :10].tolist() if hidden_states.shape[0] > 0 else None,
        )

        # CRITICAL: Match PyTorch order - patch merger BEFORE reverse window
        # PyTorch does: merger(on window-reordered patches) -> reverse window -> return
        # The merger operates on patches still in window-reordered order, then we reverse after merging
        # Patches are already grouped into 2x2 spatial blocks from window reordering,
        # so the merger can simply merge consecutive groups of 4 patches (matching PyTorch's view(-1, hidden_size))

        # Checkpoint before merger to compare inputs
        debug_checkpoint(
            "mlx_before_patch_merger_inputs",
            hidden_states_shape=hidden_states.shape,
            hidden_states_preview=hidden_states[:5, :10].tolist() if hidden_states.shape[0] > 0 else None,
        )

        # Apply patch merger on patches in window-reordered order (matching PyTorch)
        hidden_states = self.merger(hidden_states, grid_thw)

        print(f"🔎 VisionTransformer: After merger: {hidden_states.shape}")
        # Checkpoint after patch merger (critical - check if patch merger matches PyTorch)
        debug_checkpoint(
            "mlx_after_patch_merger",
            hidden_states_shape=hidden_states.shape,
            hidden_states_preview=hidden_states[:10, :10].tolist() if hidden_states.shape[0] > 0 else None,
        )

        # Now reverse window reordering (matching PyTorch's simple approach)
        # PyTorch: reverse_indices = torch.argsort(window_index); hidden_states = hidden_states[reverse_indices, :]
        # window_index is for groups (196 groups), and after merger we have 196 merged patches
        # So we can directly apply reverse_indices to the merged patches
        reverse_indices = mx.argsort(window_index.astype(mx.int32))
        hidden_states = hidden_states[reverse_indices.astype(mx.int32), :]

        # Checkpoint after reverse window (critical - check if reverse reordering matches PyTorch)
        debug_checkpoint(
            "mlx_after_reverse_window",
            skip=True,
            hidden_states_shape=hidden_states.shape,
            hidden_states_preview=hidden_states[:5, :10].tolist() if hidden_states.shape[0] > 0 else None,
        )

        return hidden_states


class QwenEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 152064,
        hidden_size: int = 3584,
        num_hidden_layers: int = 28,
        max_position_embeddings: int = 128000,
        rope_theta: float = 1000000.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.image_token_id = 151655  # <|image_pad|> token

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = []
        for i in range(num_hidden_layers):
            layer = QwenEncoderLayer()
            layer._layer_idx = i  # Set layer index for debugging
            layer.self_attn._layer_idx = i  # Set layer index on attention module too
            self.layers.append(layer)
        self.norm = QwenRMSNorm(hidden_size, eps=1e-6)
        self.rotary_emb = QwenRotaryEmbedding(
            dim=hidden_size // 28,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            rope_type="default",
        )

        # Vision transformer for VL fusion (only for Edit model, not txt2img)
        # Will be initialized when visual weights are loaded
        self.visual = None

    def get_image_features(self, pixel_values: mx.array, image_grid_thw: mx.array) -> mx.array:
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.
        Matches the reference implementation exactly.
        """
        if self.visual is None:
            raise RuntimeError("Vision transformer not initialized. Call load_visual_weights() first.")

        # pixel_values can be:
        # - Flattened: [num_patches, C*T*H*W] (from HF processor)
        # - 5D NCDHW: [num_patches, C, T, H, W] (from native patches)
        # Flatten 5D to match PyTorch's approach - patch_embed will reshape
        if len(pixel_values.shape) == 5:
            num_patches = pixel_values.shape[0]
            pixel_values = pixel_values.reshape(num_patches, -1)
            print(f"🔎 Flattened 5D input to: {pixel_values.shape}")

        print("\n🔎 MLX Vision Tower Input:")
        print(f"   pixel_values shape: {pixel_values.shape}")
        print(f"   pixel_values dtype: {pixel_values.dtype}")
        print(f"   pixel_values mean: {float(mx.mean(pixel_values)):.6f}")
        print(f"   pixel_values min: {float(mx.min(pixel_values)):.6f}")
        print(f"   pixel_values max: {float(mx.max(pixel_values)):.6f}")

        pixel_values = pixel_values.astype(mx.float32)
        image_embeds = self.visual(pixel_values, image_grid_thw)

        print("\n🔎 MLX Vision Tower Output:")
        print(f"   Shape: {image_embeds.shape}")
        print(f"   Dtype: {image_embeds.dtype}")
        print(f"   Mean: {float(mx.mean(image_embeds)):.6f}")
        print(f"   Min: {float(mx.min(image_embeds)):.6f}")
        print(f"   Max: {float(mx.max(image_embeds)):.6f}")
        print(f"   Std: {float(mx.std(image_embeds)):.6f}\n")

        # Split embeddings based on grid sizes
        # CRITICAL: After patch merger, each image's embeddings are reduced by factor of 4
        # Original grid: H*W patches per image
        # After merger: (H*W)/4 embeddings per image (spatial merge factor = 2x2 = 4)
        # So we need to divide the original grid size by 4 to get the actual split sizes
        original_split_sizes = image_grid_thw.prod(axis=-1).astype(mx.int32)
        # Account for patch merger: divide by 4 (2x2 spatial merge)
        split_sizes = (original_split_sizes // 4).astype(mx.int32)
        split_sizes = [int(s) for s in split_sizes.tolist()]
        # Filter out zero sizes (in case of empty images)
        split_sizes = [s for s in split_sizes if s > 0]

        # Manually split to ensure correct behavior (mx.split may behave differently than torch.split)
        # PyTorch's torch.split(tensor, [196, 196, 196], dim=0) creates 3 tensors of size 196 each
        # We'll manually slice to match this behavior exactly
        image_embeds_split = []
        start_idx = 0
        for split_size in split_sizes:
            end_idx = start_idx + split_size
            if end_idx <= image_embeds.shape[0]:
                image_embeds_split.append(image_embeds[start_idx:end_idx])
                start_idx = end_idx
            else:
                # If we run out of embeddings, break (shouldn't happen if sizes are correct)
                break

        # Checkpoint: Capture vision embeddings before fusion
        from mflux_debugger.semantic_checkpoint import debug_checkpoint

        # Concatenate for checkpoint (easier to compare)
        image_embeds_concat = (
            mx.concatenate(image_embeds_split, axis=0) if len(image_embeds_split) > 0 else image_embeds
        )
        debug_checkpoint(
            "mlx_vision_embeds_raw",
            {
                "image_embeds_shape": image_embeds_concat.shape,
                "image_embeds_preview": image_embeds_concat[:10, :10].tolist()
                if image_embeds_concat.shape[0] > 0
                else None,
                "num_splits": len(image_embeds_split),
                "split_sizes": split_sizes,
                "original_split_sizes": original_split_sizes.tolist(),
                "image_grid_thw": image_grid_thw.tolist(),
            },
        )

        # Checkpoint: Individual image embeddings for multi-image debugging
        if len(image_embeds_split) > 1:
            debug_checkpoint(
                "mlx_vision_embeds_split",
                {
                    "num_images": len(image_embeds_split),
                    "image1_shape": image_embeds_split[0].shape if len(image_embeds_split) > 0 else None,
                    "image1_preview": image_embeds_split[0][:5, :10].tolist()
                    if len(image_embeds_split) > 0 and image_embeds_split[0].shape[0] > 0
                    else None,
                    "image2_shape": image_embeds_split[1].shape if len(image_embeds_split) > 1 else None,
                    "image2_preview": image_embeds_split[1][:5, :10].tolist()
                    if len(image_embeds_split) > 1 and image_embeds_split[1].shape[0] > 0
                    else None,
                },
            )

        return image_embeds_split

    def get_placeholder_mask(
        self,
        input_ids: mx.array,
        inputs_embeds: mx.array,
        image_features: mx.array = None,
    ) -> mx.array:
        """
        Obtains multimodal placeholder mask from input_ids and checks that the placeholder token count
        is equal to the length of multimodal features. Matches the reference implementation.
        """
        # Find image token positions
        special_image_mask = input_ids == self.image_token_id
        n_image_tokens = mx.sum(special_image_mask).item()

        # Expand mask to match embeddings shape
        special_image_mask = mx.expand_dims(special_image_mask, axis=-1)  # [batch, seq, 1]
        special_image_mask = mx.broadcast_to(special_image_mask, inputs_embeds.shape)  # [batch, seq, hidden]

        # Validate token count matches features
        if image_features is not None:
            total_image_features = image_features.shape[0]  # image_features is already concatenated
            if n_image_tokens != total_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {total_image_features}"
                )

        return special_image_mask

    def __call__(
        self,
        input_ids: mx.array | None = None,
        attention_mask: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
        pixel_values: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
        precomputed_image_embeds: mx.array | None = None,
    ) -> mx.array:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided.")
            batch_size, seq_len = input_ids.shape
            inputs_embeds = self.embed_tokens(input_ids)

            # Checkpoint: After embedding lookup, before fusion
            from mflux_debugger.semantic_checkpoint import debug_checkpoint

            debug_checkpoint(
                "mlx_after_embed_lookup_before_fusion",
                inputs_embeds_shape=inputs_embeds.shape,
                inputs_embeds_at_vision_positions=inputs_embeds[0, 65:70, :10].tolist()
                if inputs_embeds.shape[0] > 0 and inputs_embeds.shape[1] > 65
                else None,
                seq_len=seq_len,
                input_ids_at_vision_positions=input_ids[0, 65:70].tolist()
                if input_ids.shape[0] > 0 and input_ids.shape[1] > 65
                else None,
            )
        else:
            batch_size, seq_len, _ = inputs_embeds.shape

        # Vision-Language fusion - matches reference implementation exactly
        if precomputed_image_embeds is not None and image_grid_thw is not None:
            # Use precomputed embeddings directly - bypass entire vision pipeline
            image_embeds = precomputed_image_embeds

        elif pixel_values is not None and image_grid_thw is not None:
            print(
                f"🔎 QwenEncoder: Processing VL fusion (pixel_values.shape={pixel_values.shape}, image_grid_thw={image_grid_thw})"
            )

            # Detect format: if pixel_values has 5 dimensions, it's native patches [num_patches, C, T, H, W]
            # If it has 2 dimensions, it's processed embeddings from tokenizer that we need to reject
            if len(pixel_values.shape) == 5:
                print("🔎 QwenEncoder: Native patch format detected - processing through vision transformer")
                # Native format: process through vision transformer
                image_embeds = self.get_image_features(pixel_values, image_grid_thw)
                image_embeds = mx.concatenate(image_embeds, axis=0)
            elif len(pixel_values.shape) == 2:
                print("🔎 QwenEncoder: HF processor 2D format detected - reshaping to 5D")
                print(f"🔎 QwenEncoder: Input shape: {pixel_values.shape}")

                # HF processor gives us (num_patches, 1176) where 1176 = C*T*H*W = 3*2*14*14
                # Keep it flattened - patch_embed will reshape it (matching PyTorch's approach)
                print(f"🔎 QwenEncoder: Keeping flattened format: {pixel_values.shape}")

                # Checkpoint: Right before vision transformer to compare inputs (verified - inputs match)
                from mflux_debugger.semantic_checkpoint import debug_checkpoint

                debug_checkpoint(
                    "mlx_before_vision_transformer_call",
                    skip=True,
                    pixel_values_shape=pixel_values.shape,
                    pixel_values_preview=pixel_values[:5, :10].tolist() if pixel_values.shape[0] > 0 else None,
                    pixel_values_mean=float(mx.mean(pixel_values).item()),
                    pixel_values_std=float(mx.std(pixel_values).item()),
                    image_grid_thw=image_grid_thw.tolist(),
                )

                # Process through vision transformer - patch_embed expects flattened input
                image_embeds_split = self.get_image_features(pixel_values, image_grid_thw)
                # Concatenate embeddings in order: [image1_embeds, image2_embeds, ...]
                # This order must match the order of images in the prompt
                image_embeds = mx.concatenate(image_embeds_split, axis=0)

                print(f"🔎 QwenEncoder: Vision tower output: {image_embeds.shape}")
                print(f"🔎 QwenEncoder: Number of image splits: {len(image_embeds_split)}")
                if len(image_embeds_split) > 1:
                    print(f"🔎 QwenEncoder: Split sizes: {[e.shape[0] for e in image_embeds_split]}")
            else:
                print(f"🔎 QwenEncoder: Unknown pixel_values format with shape {pixel_values.shape}")
                raise ValueError(f"Unsupported pixel_values format: {pixel_values.shape}")

            # Replace image placeholder tokens with actual image embeddings
            image_positions = input_ids == self.image_token_id
            n_image_tokens = mx.sum(image_positions).item()

            # Checkpoint: Analyze token distribution for multiple images
            from mflux_debugger.semantic_checkpoint import debug_checkpoint

            # Find token positions to understand distribution
            image_positions_list = image_positions.flatten().tolist()
            image_token_indices = [i for i, is_img in enumerate(image_positions_list) if is_img]

            debug_checkpoint(
                "mlx_image_token_analysis",
                {
                    "n_image_tokens": n_image_tokens,
                    "total_embeddings": image_embeds.shape[0],
                    "image_token_indices": image_token_indices[:20]
                    if len(image_token_indices) > 20
                    else image_token_indices,
                    "num_splits": len(image_embeds_split) if "image_embeds_split" in locals() else 0,
                    "split_sizes": [e.shape[0] for e in image_embeds_split]
                    if "image_embeds_split" in locals() and len(image_embeds_split) > 0
                    else None,
                },
            )

            print(f"🔎 QwenEncoder: Found {n_image_tokens} image tokens, have {image_embeds.shape[0]} image embeddings")

            if n_image_tokens > 0 and image_embeds.shape[0] >= n_image_tokens:
                # Replace image tokens with image embeddings
                # Create a list of embeddings and then stack them
                image_positions_flat = image_positions.flatten()
                inputs_embeds_flat = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])

                # Match PyTorch exactly: concatenate all embeddings and use sequentially
                # PyTorch does: image_embeds = torch.cat(image_embeds_split, dim=0) then masked_scatter
                # masked_scatter fills positions sequentially from the source tensor
                # PyTorch's get_placeholder_mask validates that tokens == embeddings, so they must match
                # For multi-image, PyTorch produces more tokens to match all embeddings
                # We should use all available embeddings if we have enough tokens, otherwise truncate
                # CRITICAL: For multi-image support, we need enough tokens to use all embeddings
                if image_embeds.shape[0] <= n_image_tokens:
                    # We have enough tokens - use all embeddings (matches PyTorch)
                    image_embeds_to_use = image_embeds
                    if len(image_embeds_split) > 1:
                        print(
                            f"🔎 QwenEncoder: Multi-image - using all {image_embeds.shape[0]} embeddings with {n_image_tokens} tokens"
                        )
                else:
                    # More embeddings than tokens - truncate to match token count
                    # This should only happen if tokenization was incorrect
                    image_embeds_to_use = image_embeds[:n_image_tokens]
                    print(
                        f"🔎 QwenEncoder: WARNING - More embeddings ({image_embeds.shape[0]}) than tokens ({n_image_tokens}), truncating embeddings"
                    )

                if len(image_embeds_split) > 1:
                    split_sizes = [e.shape[0] for e in image_embeds_split]
                    if image_embeds.shape[0] <= n_image_tokens:
                        print(
                            f"🔎 QwenEncoder: Multi-image: Have {split_sizes} embeddings per image, using all {image_embeds.shape[0]} embeddings"
                        )
                    else:
                        print(
                            f"🔎 QwenEncoder: Multi-image: Have {split_sizes} embeddings per image, using first {n_image_tokens} of {image_embeds.shape[0]} total"
                        )

                    # Checkpoint: Analyze how embeddings are distributed
                    from mflux_debugger.semantic_checkpoint import debug_checkpoint

                    # Calculate which embeddings from which split are being used
                    # If we have [196, 196] splits and use 237 tokens:
                    # - First 196 come from split[0] (image1)
                    # - Next 41 come from split[1] (image2)
                    split_usage = []
                    remaining = n_image_tokens
                    for i, split_size in enumerate(split_sizes):
                        used_from_split = min(split_size, remaining)
                        split_usage.append(
                            {
                                "split_idx": i,
                                "used": used_from_split,
                                "total": split_size,
                                "start_idx_in_concat": sum(split_sizes[:i]),
                                "end_idx_in_concat": sum(split_sizes[:i]) + used_from_split,
                            }
                        )
                        remaining -= used_from_split
                        if remaining <= 0:
                            break

                    debug_checkpoint(
                        "mlx_embedding_distribution",
                        {
                            "n_image_tokens": n_image_tokens,
                            "total_embeddings": image_embeds.shape[0],
                            "split_sizes": split_sizes,
                            "split_usage": split_usage,
                            "image1_embeds_preview": image_embeds_split[0][:5, :10].tolist()
                            if len(image_embeds_split) > 0 and image_embeds_split[0].shape[0] > 0
                            else None,
                            "image2_embeds_preview": image_embeds_split[1][:5, :10].tolist()
                            if len(image_embeds_split) > 1 and image_embeds_split[1].shape[0] > 0
                            else None,
                            "used_embeds_preview": image_embeds_to_use[:10, :10].tolist()
                            if image_embeds_to_use.shape[0] > 0
                            else None,
                        },
                    )

                # Checkpoint: Capture vision embeddings before fusion
                from mflux_debugger.semantic_checkpoint import debug_checkpoint

                debug_checkpoint(
                    "mlx_vision_embeds_before_fusion",
                    {
                        "image_embeds_to_use_shape": image_embeds_to_use.shape,
                        "image_embeds_to_use_preview": image_embeds_to_use[:10, :10].tolist()
                        if image_embeds_to_use.shape[0] > 0
                        else None,
                        "n_image_tokens": n_image_tokens,
                    },
                )

                # Build new embeddings list
                new_embeds_list = []
                image_idx = 0
                for i in range(len(image_positions_flat)):
                    if image_positions_flat[i] and image_idx < image_embeds_to_use.shape[0]:
                        new_embeds_list.append(image_embeds_to_use[image_idx])
                        image_idx += 1
                    else:
                        new_embeds_list.append(inputs_embeds_flat[i])

                # Stack the new embeddings
                new_embeds = mx.stack(new_embeds_list, axis=0)
                inputs_embeds = new_embeds.reshape(inputs_embeds.shape)
                print(f"🔎 QwenEncoder: Replaced {image_idx} image tokens with embeddings")

                # Checkpoint: After fusion to see what was actually placed
                if len(image_embeds_split) > 1:
                    debug_checkpoint(
                        "mlx_after_fusion_analysis",
                        {
                            "inputs_embeds_shape": inputs_embeds.shape,
                            "first_image_token_embeds": inputs_embeds[
                                0, image_token_indices[0] : image_token_indices[0] + 5, :10
                            ].tolist()
                            if len(image_token_indices) > 0
                            else None,
                            "middle_token_embeds": inputs_embeds[
                                0,
                                image_token_indices[len(image_token_indices) // 2] : image_token_indices[
                                    len(image_token_indices) // 2
                                ]
                                + 5,
                                :10,
                            ].tolist()
                            if len(image_token_indices) > len(image_token_indices) // 2
                            else None,
                            "last_image_token_embeds": inputs_embeds[
                                0, image_token_indices[-1] - 4 : image_token_indices[-1] + 1, :10
                            ].tolist()
                            if len(image_token_indices) > 0
                            else None,
                        },
                    )

                # Checkpoint: Right after fusion
                debug_checkpoint(
                    "mlx_after_fusion",
                    inputs_embeds_shape=inputs_embeds.shape,
                    inputs_embeds_at_vision_positions=inputs_embeds[0, 65:70, :10].tolist()
                    if inputs_embeds.shape[0] > 0 and inputs_embeds.shape[1] > 65
                    else None,
                    seq_len=inputs_embeds.shape[1],
                )

                # Checkpoint: Capture fused embeddings at vision token positions
                # MLX doesn't support boolean indexing, so we convert to numpy, get indices, then convert back
                if mx.any(image_positions_flat):
                    import numpy as np

                    image_positions_np = np.array(image_positions_flat)
                    vision_token_indices_np = np.where(image_positions_np)[0][:10]
                    vision_token_indices = mx.array(vision_token_indices_np)
                else:
                    vision_token_indices = mx.array([])
                if vision_token_indices.size > 0:
                    vision_embeds_at_positions = (
                        new_embeds[vision_token_indices[:5]] if len(vision_token_indices) > 0 else None
                    )
                    debug_checkpoint(
                        "mlx_vision_embeds_after_fusion",
                        {
                            "vision_embeds_at_positions_preview": vision_embeds_at_positions[:5, :10].tolist()
                            if vision_embeds_at_positions is not None and vision_embeds_at_positions.shape[0] > 0
                            else None,
                            "vision_token_indices": vision_token_indices[:10].tolist(),
                        },
                    )

        cache_position = mx.arange(seq_len, dtype=mx.int32)
        position_ids = mx.expand_dims(mx.expand_dims(cache_position, axis=0), axis=0)
        position_ids = mx.broadcast_to(position_ids, (3, batch_size, seq_len))
        if attention_mask is not None:
            padding_mask = mx.where(
                attention_mask == 1,
                mx.zeros_like(attention_mask).astype(mx.float32),
                mx.ones_like(attention_mask).astype(mx.float32) * (-float("inf")),
            )
            padding_mask = mx.expand_dims(mx.expand_dims(padding_mask, axis=1), axis=1)
        else:
            padding_mask = None

        # Create causal triangular mask
        idx = mx.arange(seq_len, dtype=mx.int32)
        j = mx.expand_dims(idx, axis=0)
        i = mx.expand_dims(idx, axis=1)
        tri_bool = j > i
        zeros_2d = mx.zeros((seq_len, seq_len)).astype(mx.float32)
        neginf_2d = mx.ones((seq_len, seq_len)).astype(mx.float32) * (-float("inf"))
        causal_tri_mask = mx.where(tri_bool, neginf_2d, zeros_2d)
        causal_tri_mask = mx.expand_dims(mx.expand_dims(causal_tri_mask, axis=0), axis=0)  # [1,1,S,S]
        causal_tri_mask = mx.broadcast_to(causal_tri_mask, (batch_size, 1, seq_len, seq_len))

        # Final attention mask: causal triangle + padding mask
        if padding_mask is not None:
            attention_mask_4d = causal_tri_mask + padding_mask
        else:
            attention_mask_4d = causal_tri_mask

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Checkpoint before first layer to compare inputs
        from mflux_debugger.semantic_checkpoint import debug_checkpoint

        debug_checkpoint(
            "mlx_before_first_text_layer",
            inputs_embeds_shape=hidden_states.shape,
            inputs_embeds_at_vision_positions=hidden_states[0, 65:70, :10].tolist()
            if hidden_states.shape[0] > 0 and hidden_states.shape[1] > 65
            else None,
            seq_len=hidden_states.shape[1],
        )

        # Transformer layers
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, attention_mask_4d, position_embeddings)

            # Debug checkpoints at key layers to find where divergence starts
            if i in [0, 1, 3, 7, 14, 21, 27]:  # Check key layers
                from mflux_debugger.semantic_checkpoint import debug_checkpoint

                debug_checkpoint(
                    f"mlx_text_encoder_after_layer_{i}",
                    hidden_states_shape=hidden_states.shape,
                    hidden_states_at_vision_positions=hidden_states[0, 65:70, :10].tolist()
                    if hidden_states.shape[0] > 0 and hidden_states.shape[1] > 65
                    else None,
                    layer_idx=i,
                )

        # Apply norm AFTER ALL layers (matching PyTorch reference)
        hidden_states = self.norm(hidden_states)

        return hidden_states
