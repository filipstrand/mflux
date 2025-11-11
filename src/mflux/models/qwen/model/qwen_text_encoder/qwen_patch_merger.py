import mlx.core as mx
from mlx import nn


class PatchMerger(nn.Module):
    def __init__(self, context_dim: int, hidden_size: int, spatial_merge_size: int = 2):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.hidden_size_merged = context_dim * (spatial_merge_size**2)
        self.ln_q = nn.RMSNorm(context_dim, eps=1e-6)
        self.mlp_0 = nn.Linear(self.hidden_size_merged, self.hidden_size_merged, bias=True)
        self.mlp_1 = nn.Linear(self.hidden_size_merged, hidden_size, bias=True)

    def __call__(self, x: mx.array, grid_thw: mx.array) -> mx.array:
        if not hasattr(self, "_weights_logged"):
            self._weights_logged = True
        x = self.ln_q(x)
        merged_patches = []
        offset = 0
        for t, h, w in grid_thw:
            t, h, w = int(t), int(h), int(w)
            num_patches_this_image = t * h * w
            x_this_image = x[offset : offset + num_patches_this_image]
            x_merged = x_this_image.reshape(-1, self.hidden_size_merged)
            merged_patches.append(x_merged)
            offset += num_patches_this_image

        x = mx.concatenate(merged_patches, axis=0)
        x = self.mlp_0(x)
        x = nn.gelu(x)
        x = self.mlp_1(x)
        return x
