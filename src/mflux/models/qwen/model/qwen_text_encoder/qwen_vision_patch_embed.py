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

        self.proj = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=[temporal_patch_size, patch_size, patch_size],
            stride=[temporal_patch_size, patch_size, patch_size],
            bias=False,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(
            batch_size, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = hidden_states.transpose(0, 2, 3, 4, 1)
        output = self.proj(hidden_states)
        return output.reshape(batch_size, self.embed_dim)
