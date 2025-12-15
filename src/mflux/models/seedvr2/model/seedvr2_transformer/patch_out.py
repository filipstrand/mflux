import mlx.core as mx
from mlx import nn


class PatchOut(nn.Module):
    def __init__(
        self,
        out_channels: int = 16,
        patch_size: tuple = (1, 2, 2),
        dim: int = 2560,
    ):
        super().__init__()
        self.patch_size = patch_size
        t, h, w = patch_size
        self.proj = nn.Linear(dim, out_channels * t * h * w)

    def __call__(self, vid: mx.array, vid_shape: mx.array) -> tuple[mx.array, mx.array]:
        t, h, w = self.patch_size
        vid = self.proj(vid)

        B = vid.shape[0]
        T_patches = int(vid_shape[0, 0])
        H_patches = int(vid_shape[0, 1])
        W_patches = int(vid_shape[0, 2])
        C = vid.shape[-1] // (t * h * w)

        vid = vid.reshape(B, T_patches, H_patches, W_patches, t, h, w, C)
        vid = vid.transpose(0, 7, 1, 4, 2, 5, 3, 6)
        vid = vid.reshape(B, C, T_patches * t, H_patches * h, W_patches * w)

        return vid, vid_shape
