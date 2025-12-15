import mlx.core as mx
from mlx import nn


class PatchIn(nn.Module):
    def __init__(
        self,
        in_channels: int = 33,
        patch_size: tuple = (1, 2, 2),
        dim: int = 2560,
    ):
        super().__init__()
        self.patch_size = patch_size
        t, h, w = patch_size
        self.proj = nn.Linear(in_channels * t * h * w, dim)

    def __call__(self, vid: mx.array) -> tuple[mx.array, mx.array]:
        t, h, w = self.patch_size
        B, C, T, H, W = vid.shape

        T_patches = T // t
        H_patches = H // h
        W_patches = W // w

        vid = vid.reshape(B, C, T_patches, t, H_patches, h, W_patches, w)
        vid = vid.transpose(0, 2, 4, 6, 3, 5, 7, 1)
        vid = vid.reshape(B, T_patches, H_patches, W_patches, t * h * w * C)

        vid = self.proj(vid)
        vid = vid.reshape(B, -1, vid.shape[-1])
        vid_shape = mx.broadcast_to(mx.array([T_patches, H_patches, W_patches], dtype=mx.int32), (B, 3))

        return vid, vid_shape
