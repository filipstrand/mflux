from __future__ import annotations

import mlx.core as mx
import numpy as np
from mlx import nn


class QwenEmbedRopeMLX(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int], scale_rope: bool = False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.scale_rope = scale_rope

        pos_index = np.arange(4096, dtype=np.int32)
        neg_index = (np.arange(4096, dtype=np.int32)[::-1] * -1) - 1

        self.pos_freqs = np.concatenate(
            [
                self._rope_params(pos_index, self.axes_dim[0], self.theta),
                self._rope_params(pos_index, self.axes_dim[1], self.theta),
                self._rope_params(pos_index, self.axes_dim[2], self.theta),
            ],
            axis=1,
        )
        self.neg_freqs = np.concatenate(
            [
                self._rope_params(neg_index, self.axes_dim[0], self.theta),
                self._rope_params(neg_index, self.axes_dim[1], self.theta),
                self._rope_params(neg_index, self.axes_dim[2], self.theta),
            ],
            axis=1,
        )

    def _rope_params(self, index: np.ndarray, dim: int, theta: int) -> np.ndarray:
        assert dim % 2 == 0
        scales = np.arange(0, dim, 2, dtype=np.float32) / dim
        omega = 1.0 / (theta**scales)
        freqs = np.outer(index.astype(np.float32), omega)

        cos_freqs = np.cos(freqs)
        sin_freqs = np.sin(freqs)

        return np.stack([cos_freqs, sin_freqs], axis=-1)

    def _compute_video_freqs(self, frame: int, height: int, width: int, idx: int = 0) -> tuple[np.ndarray, np.ndarray]:
        seq_lens = frame * height * width

        axes_splits = [x // 2 for x in self.axes_dim]
        freqs_pos = np.split(self.pos_freqs, np.cumsum(axes_splits)[:-1], axis=1)
        freqs_neg = np.split(self.neg_freqs, np.cumsum(axes_splits)[:-1], axis=1)

        freqs_frame_raw = freqs_pos[0][idx : idx + frame]
        freqs_frame = freqs_frame_raw.reshape(frame, 1, 1, -1, 2)
        freqs_frame = np.broadcast_to(freqs_frame, (frame, height, width, freqs_frame.shape[-2], 2))

        if self.scale_rope:
            freqs_height = np.concatenate(
                [freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], axis=0
            )
        else:
            freqs_height = freqs_pos[1][:height]
        freqs_height = freqs_height.reshape(1, height, 1, -1, 2)
        freqs_height = np.broadcast_to(freqs_height, (frame, height, width, freqs_height.shape[-2], 2))

        if self.scale_rope:
            freqs_width = np.concatenate([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], axis=0)
        else:
            freqs_width = freqs_pos[2][:width]
        freqs_width = freqs_width.reshape(1, 1, width, -1, 2)
        freqs_width = np.broadcast_to(freqs_width, (frame, height, width, freqs_width.shape[-2], 2))

        freqs = np.concatenate([freqs_frame, freqs_height, freqs_width], axis=-2)
        freqs = freqs.reshape(seq_lens, -1, 2)

        cos_freqs = freqs[..., 0]
        sin_freqs = freqs[..., 1]

        return cos_freqs, sin_freqs

    def __call__(
        self,
        video_fhw: tuple[int, int, int] | list[tuple[int, int, int]],
        txt_seq_lens: list[int],
    ) -> tuple[tuple[mx.array, mx.array], tuple[mx.array, mx.array]]:
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_cos_list = []
        vid_sin_list = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            cos_v, sin_v = self._compute_video_freqs(frame, height, width, idx)
            vid_cos_list.append(cos_v)
            vid_sin_list.append(sin_v)

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        vid_cos = np.concatenate(vid_cos_list, axis=0)
        vid_sin = np.concatenate(vid_sin_list, axis=0)

        max_len = max(txt_seq_lens)
        txt_cos = self.pos_freqs[max_vid_index : max_vid_index + max_len, :, 0]
        txt_sin = self.pos_freqs[max_vid_index : max_vid_index + max_len, :, 1]

        return (
            (mx.array(vid_cos.astype(np.float32)), mx.array(vid_sin.astype(np.float32))),
            (mx.array(txt_cos.astype(np.float32)), mx.array(txt_sin.astype(np.float32))),
        )
