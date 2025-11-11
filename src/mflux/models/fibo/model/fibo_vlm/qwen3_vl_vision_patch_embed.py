"""Qwen3VL Vision Patch Embedding for MLX."""

import mlx.core as mx
from mlx import nn


class Qwen3VLVisionPatchEmbed(nn.Module):
    """Patch embedding layer for Qwen3VL vision transformer."""

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

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        stride = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            bias=True,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """
        Args:
            hidden_states: Shape (seq_len, in_channels * temporal_patch_size * patch_size * patch_size)
                          Flattened patches from processor
        Returns:
            embedded: Shape (seq_len, embed_dim)
        """
        # Reshape to match PyTorch: (seq_len, in_channels, temporal_patch_size, patch_size, patch_size)
        # Input is flattened: (seq_len, in_channels * temporal_patch_size * patch_size * patch_size)
        seq_len = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(
            seq_len,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )

        # MLX Conv3d expects channels-last format: (batch, depth, height, width, channels)
        # Transpose from (seq_len, in_channels, temporal_patch_size, patch_size, patch_size)
        #            to (seq_len, temporal_patch_size, patch_size, patch_size, in_channels)
        hidden_states = hidden_states.transpose(0, 2, 3, 4, 1)

        # Apply conv3d
        # With stride=kernel_size, output shape is (seq_len, 1, 1, 1, embed_dim)
        output = self.proj(hidden_states)

        # Reshape to (seq_len, embed_dim)
        # Output from conv3d: (seq_len, 1, 1, 1, embed_dim) -> squeeze spatial dims -> (seq_len, embed_dim)
        output = output.reshape(seq_len, self.embed_dim)
        return output
