from __future__ import annotations

import mlx.core as mx
import numpy as np
from mlx import nn


class QwenEmbedRopeMLX(nn.Module):
    """
    Faithful MLX port of QwenEmbedRope from diffusers.

    This implementation closely matches the PyTorch reference to ensure numerical consistency.
    Key differences from PyTorch:
    - Returns (cos, sin) tuples instead of complex tensors (MLX attention expects this format)
    - Uses NumPy for caching to avoid MLX graph issues with buffers
    """

    def __init__(self, theta: int, axes_dim: list[int], scale_rope: bool = False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.scale_rope = scale_rope

        # Match PyTorch: cache up to 4096 positions
        pos_index = np.arange(4096, dtype=np.int32)
        neg_index = (np.arange(4096, dtype=np.int32)[::-1] * -1) - 1

        # Precompute frequencies for all 3 axes (frame, height, width)
        # Match PyTorch lines 174-189
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
        """
        Match PyTorch's rope_params (lines 194-202).

        PyTorch computes:
            freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).float().div(dim)))
            freqs = torch.polar(torch.ones_like(freqs), freqs)

        torch.polar(r, theta) creates complex number r * e^(i*theta) = r * (cos(theta) + i*sin(theta))
        Since r=1, this gives us cos(freqs) + i*sin(freqs)

        MLX doesn't have complex tensors, so we compute cos/sin separately and return them.
        """
        assert dim % 2 == 0
        # Compute frequency basis
        scales = np.arange(0, dim, 2, dtype=np.float32) / dim
        omega = 1.0 / (theta**scales)
        freqs = np.outer(index.astype(np.float32), omega)

        # PyTorch's torch.polar(1, freqs) creates complex exponentials
        # We return cos (real) and sin (imaginary) as separate arrays stacked in last dim
        # Shape: [len(index), dim//2, 2] where [..., 0] is cos and [..., 1] is sin
        cos_freqs = np.cos(freqs)
        sin_freqs = np.sin(freqs)

        # Stack as [len(index), dim//2, 2] to match complex tensor structure
        return np.stack([cos_freqs, sin_freqs], axis=-1)

    def _compute_video_freqs(self, frame: int, height: int, width: int, idx: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """
        Match PyTorch's _compute_video_freqs (lines 249-265).

        Returns:
            (cos_freqs, sin_freqs): Tuple of [seq_len, total_dim//2] arrays
        """
        seq_lens = frame * height * width

        # Split pos/neg freqs by axes dimensions
        # Match PyTorch line 251
        axes_splits = [x // 2 for x in self.axes_dim]
        freqs_pos = np.split(self.pos_freqs, np.cumsum(axes_splits)[:-1], axis=1)
        freqs_neg = np.split(self.neg_freqs, np.cumsum(axes_splits)[:-1], axis=1)

        # Frame frequencies (match PyTorch line 253)
        # PyTorch: freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        # PyTorch's complex tensor has shape [frame, 1, 1, axes_dim[0]//2] -> [frame, height, width, axes_dim[0]//2]
        # MLX: we have [frame, axes_dim[0]//2, 2] (cos/sin), need [frame, height, width, axes_dim[0]//2, 2]
        freqs_frame_raw = freqs_pos[0][idx : idx + frame]  # [frame, axes_dim[0]//2, 2]
        freqs_frame = freqs_frame_raw.reshape(frame, 1, 1, -1, 2)  # [frame, 1, 1, axes_dim[0]//2, 2]
        freqs_frame = np.broadcast_to(freqs_frame, (frame, height, width, freqs_frame.shape[-2], 2))

        # Height frequencies (match PyTorch lines 255-262)
        if self.scale_rope:
            # PyTorch: freqs_height = torch.cat([freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0)
            freqs_height = np.concatenate(
                [freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], axis=0
            )
        else:
            # PyTorch: freqs_height = freqs_pos[1][:height]
            freqs_height = freqs_pos[1][:height]
        # PyTorch: freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
        freqs_height = freqs_height.reshape(1, height, 1, -1, 2)  # [1, height, 1, axes_dim[1]//2, 2]
        freqs_height = np.broadcast_to(freqs_height, (frame, height, width, freqs_height.shape[-2], 2))

        # Width frequencies (match PyTorch lines 255-262)
        if self.scale_rope:
            # PyTorch: freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
            freqs_width = np.concatenate([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], axis=0)
        else:
            # PyTorch: freqs_width = freqs_pos[2][:width]
            freqs_width = freqs_pos[2][:width]
        # PyTorch: freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        freqs_width = freqs_width.reshape(1, 1, width, -1, 2)  # [1, 1, width, axes_dim[2]//2, 2]
        freqs_width = np.broadcast_to(freqs_width, (frame, height, width, freqs_width.shape[-2], 2))

        # Concatenate all frequencies (match PyTorch line 263)
        # PyTorch: freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1)
        # PyTorch concatenates along the last dimension (complex dimension)
        # MLX: concatenate along axis=-2 (the dimension before the cos/sin dimension)
        freqs = np.concatenate(
            [freqs_frame, freqs_height, freqs_width], axis=-2
        )  # [frame, height, width, total_dim//2, 2]
        # PyTorch: freqs.reshape(seq_lens, -1)
        # MLX: reshape to [seq_lens, total_dim//2, 2] then split
        freqs = freqs.reshape(seq_lens, -1, 2)  # [seq_lens, total_dim//2, 2]

        # Split into cos and sin components
        cos_freqs = freqs[..., 0]  # [seq_lens, total_dim//2]
        sin_freqs = freqs[..., 1]  # [seq_lens, total_dim//2]

        return cos_freqs, sin_freqs

    def __call__(
        self,
        video_fhw: tuple[int, int, int] | list[tuple[int, int, int]],
        txt_seq_lens: list[int],
    ) -> tuple[tuple[mx.array, mx.array], tuple[mx.array, mx.array]]:
        """
        Match PyTorch's forward method (lines 204-246).

        Args:
            video_fhw: Single (F, H, W) tuple or list of tuples for multiple images
            txt_seq_lens: List of text sequence lengths (one per batch item)

        Returns:
            ((img_cos, img_sin), (txt_cos, txt_sin)): RoPE embeddings as MLX arrays
        """
        # Normalize input to list of shapes
        # Note: PyTorch has lines 223-226 that handle nested batch structure [[shape1, shape2]]
        # MLX passes shapes directly as [shape1, shape2], so we just ensure it's a list
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        # Compute video frequencies (match PyTorch lines 228-240)
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

        # Concatenate all video frequencies (match PyTorch line 244)
        vid_cos = np.concatenate(vid_cos_list, axis=0)
        vid_sin = np.concatenate(vid_sin_list, axis=0)

        # Text frequencies (match PyTorch lines 242-243)
        max_len = max(txt_seq_lens)
        txt_cos = self.pos_freqs[max_vid_index : max_vid_index + max_len, :, 0]
        txt_sin = self.pos_freqs[max_vid_index : max_vid_index + max_len, :, 1]

        # Convert to MLX arrays and return
        # Format: ((img_cos, img_sin), (txt_cos, txt_sin))
        return (
            (mx.array(vid_cos.astype(np.float32)), mx.array(vid_sin.astype(np.float32))),
            (mx.array(txt_cos.astype(np.float32)), mx.array(txt_sin.astype(np.float32))),
        )
