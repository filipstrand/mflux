import mlx.core as mx
import numpy as np
from mlx import nn

from mflux.models.qwen.model.qwen_text_encoder.qwen_patch_merger import PatchMerger
from mflux.models.qwen.model.qwen_text_encoder.qwen_vision_block import VisionBlock
from mflux.models.qwen.model.qwen_text_encoder.qwen_vision_patch_embed import VisionPatchEmbed
from mflux.models.qwen.model.qwen_text_encoder.qwen_vision_rotary_embedding import VisionRotaryEmbedding


class VisionTransformer(nn.Module):
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

        # Patch embedding (will reshape internally to match PyTorch's NCDHW format)
        hidden_states = self.patch_embed(pixel_values)  # [num_patches, embed_dim]

        # Rotary position embeddings (with spatial merging order)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)  # [num_patches, dim] where dim = head_dim

        # Get window index and cu_seqlens for windowed attention
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)

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
        # window_index is for GROUPS (after grouping), not individual patches
        # PyTorch groups consecutive patches: reshape(seq_len // spatial_merge_unit, spatial_merge_unit, -1)
        # This means groups are: [0,1,2,3], [4,5,6,7], [8,9,10,11], ... (consecutive patches in row-major order)
        # NOT 2x2 spatial blocks! The window_index expects this consecutive grouping.
        seq_len = hidden_states.shape[0]
        num_groups = seq_len // self.spatial_merge_unit
        hidden_states_grouped = hidden_states.reshape(
            num_groups, self.spatial_merge_unit, -1
        )  # [total_groups, spatial_merge_unit, hidden_dim]

        # Now reorder groups using window_index
        hidden_states_grouped = hidden_states_grouped[window_index.astype(mx.int32), :, :]

        # Reshape back to individual patches: [total_groups, spatial_merge_unit, hidden_dim] -> [total_patches, hidden_dim]
        hidden_states = hidden_states_grouped.reshape(seq_len, -1)

        # Match PyTorch's simple consecutive grouping for rotary embeddings too
        # Same as hidden_states - group consecutive patches
        rotary_pos_emb_grouped = rotary_pos_emb.reshape(
            num_groups, self.spatial_merge_unit, -1
        )  # [total_groups, spatial_merge_unit, hidden_dim]

        # Now reorder groups using window_index
        rotary_pos_emb_grouped = rotary_pos_emb_grouped[window_index.astype(mx.int32), :, :]

        # Reshape back to individual patches: [total_groups, spatial_merge_unit, hidden_dim] -> [total_patches, hidden_dim]
        rotary_pos_emb = rotary_pos_emb_grouped.reshape(seq_len, -1)

        # Duplicate embeddings (matching HF: torch.cat((emb, emb), dim=-1))
        emb = mx.concatenate([rotary_pos_emb, rotary_pos_emb], axis=-1)  # [num_patches, dim*2]
        position_embeddings = (mx.cos(emb), mx.sin(emb))

        # Apply vision transformer blocks with windowed/full attention switching
        for layer_num, block in enumerate(self.blocks):
            # Choose cu_seqlens based on whether this is a full attention layer
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens

            hidden_states = block(hidden_states, position_embeddings, cu_seqlens_now)

        hidden_states = self.merger(hidden_states, grid_thw)

        # Reverse window reordering (matching PyTorch's simple approach)
        reverse_indices = mx.argsort(window_index.astype(mx.int32))
        hidden_states = hidden_states[reverse_indices.astype(mx.int32), :]

        return hidden_states
