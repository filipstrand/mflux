from __future__ import annotations

import mlx.core as mx
import numpy as np
from mlx import nn


class QwenEmbedRopeMLX(nn.Module):
    """
    Faithful MLX port of Qwen's RoPE embedding helper used in the flux_transformer.

    Exposes a forward(video_fhw, txt_seq_lens) API that returns rotation matrices
    for image and text streams with shape [1, 1, S, D/2, 2, 2].
    """

    def __init__(self, theta: int, axes_dim: list[int], scale_rope: bool = True):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.scale_rope = scale_rope

        # Precompute positive and negative index caches up to 1024 as in reference
        pos_index = np.arange(1024, dtype=np.int32)
        neg_index = (np.arange(1024, dtype=np.int32)[::-1] * -1) - 1

        self._pos_cos = []
        self._pos_sin = []
        self._neg_cos = []
        self._neg_sin = []
        for dim in axes_dim:
            cos_p, sin_p = self._rope_params(pos_index, dim)
            cos_n, sin_n = self._rope_params(neg_index, dim)
            self._pos_cos.append(cos_p)
            self._pos_sin.append(sin_p)
            self._neg_cos.append(cos_n)
            self._neg_sin.append(sin_n)

    @staticmethod
    def _rope_params(index: np.ndarray, dim: int, theta: int = 10000) -> tuple[np.ndarray, np.ndarray]:
        assert dim % 2 == 0
        scales = np.arange(0, dim, 2, dtype=np.float32) / dim
        omega = 1.0 / (theta**scales)
        out = np.outer(index.astype(np.float32), omega).astype(np.float32)
        return np.cos(out), np.sin(out)

    def _build_video_freqs(self, frame: int, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
        dim_f, dim_h, dim_w = self.axes_dim

        cos_f = self._pos_cos[0][:frame].reshape(frame, 1, 1, -1)
        sin_f = self._pos_sin[0][:frame].reshape(frame, 1, 1, -1)

        if self.scale_rope:
            cos_h = np.concatenate(
                [self._neg_cos[1][-(height - height // 2) :], self._pos_cos[1][: height // 2]], axis=0
            )
            sin_h = np.concatenate(
                [self._neg_sin[1][-(height - height // 2) :], self._pos_sin[1][: height // 2]], axis=0
            )
        else:
            cos_h = self._pos_cos[1][:height]
            sin_h = self._pos_sin[1][:height]
        cos_h = cos_h.reshape(1, height, 1, -1)
        sin_h = sin_h.reshape(1, height, 1, -1)

        if self.scale_rope:
            cos_w = np.concatenate([self._neg_cos[2][-(width - width // 2) :], self._pos_cos[2][: width // 2]], axis=0)
            sin_w = np.concatenate([self._neg_sin[2][-(width - width // 2) :], self._pos_sin[2][: width // 2]], axis=0)
        else:
            cos_w = self._pos_cos[2][:width]
            sin_w = self._pos_sin[2][:width]
        cos_w = cos_w.reshape(1, 1, width, -1)
        sin_w = sin_w.reshape(1, 1, width, -1)

        cos = np.concatenate(
            [
                np.broadcast_to(cos_f, (frame, height, width, cos_f.shape[-1])),
                np.broadcast_to(cos_h, (frame, height, width, cos_h.shape[-1])),
                np.broadcast_to(cos_w, (frame, height, width, cos_w.shape[-1])),
            ],
            axis=-1,
        )
        sin = np.concatenate(
            [
                np.broadcast_to(sin_f, (frame, height, width, sin_f.shape[-1])),
                np.broadcast_to(sin_h, (frame, height, width, sin_h.shape[-1])),
                np.broadcast_to(sin_w, (frame, height, width, sin_w.shape[-1])),
            ],
            axis=-1,
        )

        # Flatten to [S, D/2]
        cos = cos.reshape(-1, cos.shape[-1])
        sin = sin.reshape(-1, sin.shape[-1])
        return cos, sin

    def __call__(self, video_fhw: tuple[int, int, int] | list[tuple[int, int, int]], txt_seq_lens: list[int]):
        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        frame, height, width = video_fhw

        cos_v, sin_v = self._build_video_freqs(frame, height, width)

        # Build text freqs using max_vid_index rule
        max_vid_index = max(height // 2, width // 2) if self.scale_rope else max(height, width)
        txt_len = max(txt_seq_lens)
        # Combine axes cos/sin for positions starting at max_vid_index
        cos_full = np.concatenate(self._pos_cos, axis=1)  # [1024, 128]
        sin_full = np.concatenate(self._pos_sin, axis=1)
        cos_t = cos_full[max_vid_index : max_vid_index + txt_len]
        sin_t = sin_full[max_vid_index : max_vid_index + txt_len]

        def to_rot(cos: np.ndarray, sin: np.ndarray) -> mx.array:
            row0 = np.stack([cos, -sin], axis=-1)
            row1 = np.stack([sin, cos], axis=-1)
            rot = np.stack([row0, row1], axis=-2)
            rot = rot[None, None, ...].astype(np.float32)
            return mx.array(rot)

        return to_rot(cos_v, sin_v), to_rot(cos_t, sin_t)
