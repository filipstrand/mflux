import mlx.core as mx
from mlx import nn


class VisionPatchEmbed(nn.Module):
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
