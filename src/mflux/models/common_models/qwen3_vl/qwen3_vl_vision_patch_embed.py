import mlx.core as mx
from mlx import nn


class Qwen3VLVisionPatchEmbed(nn.Module):
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
        seq_len = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(
            seq_len,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = hidden_states.transpose(0, 2, 3, 4, 1)
        output = self.proj(hidden_states)
        output = output.reshape(seq_len, self.embed_dim)
        return output
