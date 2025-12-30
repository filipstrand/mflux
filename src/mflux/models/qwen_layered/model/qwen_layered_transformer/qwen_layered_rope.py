from __future__ import annotations

import mlx.core as mx
import numpy as np
from mlx import nn


class QwenLayeredRoPE(nn.Module):
    """
    Layer3D RoPE for Qwen-Image-Layered.
    
    Extends the base RoPE to include a layer dimension:
    - Position = (layer_index, height, width)
    - Input condition image: layer_index = -1
    - Output layers: layer_index = 0, 1, ..., N-1
    
    axes_dim = [16, 56, 56] -> [layer, height, width]
    """

    def __init__(self, theta: int = 10000, axes_dim: list[int] = None, scale_rope: bool = True):
        super().__init__()
        if axes_dim is None:
            axes_dim = [16, 56, 56]
        self.theta = theta
        self.axes_dim = axes_dim  # [layer_dim, height_dim, width_dim]
        self.scale_rope = scale_rope

        # Pre-compute frequency tables for positive and negative indices
        pos_index = np.arange(4096, dtype=np.int32)
        neg_index = (np.arange(4096, dtype=np.int32)[::-1] * -1) - 1

        # Frequency tables for each dimension
        self.pos_freqs = np.concatenate(
            [
                self._rope_params(pos_index, self.axes_dim[0], self.theta),  # Layer
                self._rope_params(pos_index, self.axes_dim[1], self.theta),  # Height
                self._rope_params(pos_index, self.axes_dim[2], self.theta),  # Width
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

    def _compute_layer_freqs(
        self,
        num_layers: int,
        height: int,
        width: int,
        include_cond_image: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute 3D positional frequencies for layered output.
        
        Args:
            num_layers: Number of output layers (N)
            height: Latent height
            width: Latent width
            include_cond_image: Whether to include condition image at layer=-1
        """
        axes_splits = [x // 2 for x in self.axes_dim]
        freqs_pos = np.split(self.pos_freqs, np.cumsum(axes_splits)[:-1], axis=1)
        freqs_neg = np.split(self.neg_freqs, np.cumsum(axes_splits)[:-1], axis=1)

        all_cos = []
        all_sin = []

        # Condition image at layer=-1 (if included)
        if include_cond_image:
            cond_layer_cos, cond_layer_sin = self._compute_single_layer_freqs(
                layer_idx=-1,
                height=height,
                width=width,
                freqs_pos=freqs_pos,
                freqs_neg=freqs_neg,
            )
            all_cos.append(cond_layer_cos)
            all_sin.append(cond_layer_sin)

        # Output layers at layer=0..N-1
        for layer_idx in range(num_layers):
            layer_cos, layer_sin = self._compute_single_layer_freqs(
                layer_idx=layer_idx,
                height=height,
                width=width,
                freqs_pos=freqs_pos,
                freqs_neg=freqs_neg,
            )
            all_cos.append(layer_cos)
            all_sin.append(layer_sin)

        # Concatenate all layers
        img_cos = np.concatenate(all_cos, axis=0)
        img_sin = np.concatenate(all_sin, axis=0)

        return img_cos, img_sin

    def _compute_single_layer_freqs(
        self,
        layer_idx: int,
        height: int,
        width: int,
        freqs_pos: list[np.ndarray],
        freqs_neg: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute frequencies for a single layer."""
        seq_len = height * width

        # Layer dimension - single index
        if layer_idx >= 0:
            freqs_layer = freqs_pos[0][layer_idx:layer_idx + 1]
        else:
            # Negative index for condition image
            freqs_layer = freqs_neg[0][abs(layer_idx) - 1:abs(layer_idx)]
        freqs_layer = np.broadcast_to(freqs_layer, (seq_len, freqs_layer.shape[-2], 2))

        # Height dimension with optional scaling
        if self.scale_rope:
            freqs_height = np.concatenate(
                [freqs_neg[1][-(height - height // 2):], freqs_pos[1][:height // 2]], axis=0
            )
        else:
            freqs_height = freqs_pos[1][:height]
        freqs_height = freqs_height.reshape(height, 1, -1, 2)
        freqs_height = np.broadcast_to(freqs_height, (height, width, freqs_height.shape[-2], 2))
        freqs_height = freqs_height.reshape(seq_len, -1, 2)

        # Width dimension with optional scaling
        if self.scale_rope:
            freqs_width = np.concatenate(
                [freqs_neg[2][-(width - width // 2):], freqs_pos[2][:width // 2]], axis=0
            )
        else:
            freqs_width = freqs_pos[2][:width]
        freqs_width = freqs_width.reshape(1, width, -1, 2)
        freqs_width = np.broadcast_to(freqs_width, (height, width, freqs_width.shape[-2], 2))
        freqs_width = freqs_width.reshape(seq_len, -1, 2)

        # Concatenate all dimensions
        freqs = np.concatenate([freqs_layer, freqs_height, freqs_width], axis=-2)

        cos_freqs = freqs[..., 0]
        sin_freqs = freqs[..., 1]

        return cos_freqs, sin_freqs

    def __call__(
        self,
        num_layers: int,
        height: int,
        width: int,
        txt_seq_lens: list[int],
        include_cond_image: bool = True,
    ) -> tuple[tuple[mx.array, mx.array], tuple[mx.array, mx.array]]:
        """
        Compute rotary embeddings for layered output.
        
        Args:
            num_layers: Number of output layers
            height: Latent height (H/16)
            width: Latent width (W/16)
            txt_seq_lens: List of text sequence lengths per batch
            include_cond_image: Whether to include condition image embeddings
            
        Returns:
            Tuple of (image_rotary_emb, text_rotary_emb)
            Each contains (cos, sin) arrays
        """
        # Compute image frequencies for all layers
        img_cos, img_sin = self._compute_layer_freqs(
            num_layers=num_layers,
            height=height,
            width=width,
            include_cond_image=include_cond_image,
        )

        # Compute text frequencies
        if self.scale_rope:
            max_vid_index = max(height // 2, width // 2)
        else:
            max_vid_index = max(height, width)

        max_len = max(txt_seq_lens)
        txt_cos = self.pos_freqs[max_vid_index:max_vid_index + max_len, :, 0]
        txt_sin = self.pos_freqs[max_vid_index:max_vid_index + max_len, :, 1]

        return (
            (mx.array(img_cos.astype(np.float32)), mx.array(img_sin.astype(np.float32))),
            (mx.array(txt_cos.astype(np.float32)), mx.array(txt_sin.astype(np.float32))),
        )
