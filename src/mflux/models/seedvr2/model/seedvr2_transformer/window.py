import math
from typing import Callable

import mlx.core as mx


class WindowPartitioner:
    def __init__(
        self,
        shape: mx.array,
        window_size: tuple[int, int, int],
        shift: bool = False,
    ):
        self.forward_idx, self.reverse_idx, self.window_shapes, self.window_counts = (
            WindowPartitioner._create_window_indices(
                shape, lambda size: WindowPartitioner._make_windows(size, window_size, shift)
            )
        )

    def partition(self, tensor: mx.array) -> mx.array:
        return tensor[self.forward_idx]

    def reverse(self, tensor: mx.array) -> mx.array:
        return tensor[self.reverse_idx]

    @staticmethod
    def _make_windows(
        size: tuple[int, int, int],
        num_windows: tuple[int, int, int],
        shift: bool = False,
    ) -> list[tuple[slice, slice, slice]]:
        t, h, w = size
        resized_nt, resized_nh, resized_nw = num_windows

        scale = math.sqrt((45 * 80) / (h * w))
        resized_h, resized_w = round(h * scale), round(w * scale)

        wh = math.ceil(resized_h / resized_nh)
        ww = math.ceil(resized_w / resized_nw)
        wt = math.ceil(min(t, 30) / resized_nt)

        if shift:
            st = 0.5 if wt < t else 0
            sh = 0.5 if wh < h else 0
            sw = 0.5 if ww < w else 0
            nt = math.ceil((t - st) / wt) + 1 if st > 0 else 1
            nh = math.ceil((h - sh) / wh) + 1 if sh > 0 else 1
            nw = math.ceil((w - sw) / ww) + 1 if sw > 0 else 1
        else:
            st = sh = sw = 0
            nt = math.ceil(t / wt)
            nh = math.ceil(h / wh)
            nw = math.ceil(w / ww)

        windows = []
        for iw in range(nw):
            w_start = max(int((iw - sw) * ww), 0)
            w_end = min(int((iw - sw + 1) * ww), w)
            if w_end <= w_start:
                continue
            for ih in range(nh):
                h_start = max(int((ih - sh) * wh), 0)
                h_end = min(int((ih - sh + 1) * wh), h)
                if h_end <= h_start:
                    continue
                for it in range(nt):
                    t_start = max(int((it - st) * wt), 0)
                    t_end = min(int((it - st + 1) * wt), t)
                    if t_end <= t_start:
                        continue
                    windows.append((slice(t_start, t_end), slice(h_start, h_end), slice(w_start, w_end)))

        return windows

    @staticmethod
    def _flatten_list(tensors: list[mx.array]) -> tuple[mx.array, mx.array]:
        assert len(tensors) > 0
        shapes = mx.array([x.shape[:-1] for x in tensors], dtype=mx.int32)
        result = mx.concatenate([x.reshape(-1, x.shape[-1]) for x in tensors], axis=0)
        return result, shapes

    @staticmethod
    def _unflatten_list(
        tensor: mx.array,
        shapes: mx.array,
    ) -> list[mx.array]:
        lengths = mx.prod(shapes, axis=1).tolist()
        indices = mx.cumsum(mx.array(lengths[:-1])).tolist()
        pieces = mx.split(tensor, indices)
        return [p.reshape(*s.tolist(), -1) for p, s in zip(pieces, shapes)]

    @classmethod
    def _window_partition(
        cls,
        tensor: mx.array,
        shape: mx.array,
        window_fn: Callable,
    ) -> tuple[mx.array, mx.array, list[int]]:
        unflattened = WindowPartitioner._unflatten_list(tensor, shape)

        windowed = []
        window_counts = []
        for x in unflattened:
            t, h, w = x.shape[:-1]
            slices = window_fn((t, h, w))
            window_counts.append(len(slices))
            for st, sh, sw in slices:
                window = x[st, sh, sw]
                windowed.append(window)

        result, result_shape = WindowPartitioner._flatten_list(windowed)
        return result, result_shape, window_counts

    @classmethod
    def _create_window_indices(
        cls,
        shape: mx.array,
        window_fn: Callable,
    ) -> tuple[mx.array, mx.array, mx.array, list[int]]:
        total_len = int(mx.sum(mx.prod(shape, axis=1)))
        idx = mx.arange(total_len).reshape(-1, 1)
        windowed_idx, window_shapes, window_counts = WindowPartitioner._window_partition(idx, shape, window_fn)
        target_idx = windowed_idx.reshape(-1)
        reverse_idx = mx.argsort(target_idx)
        return target_idx, reverse_idx, window_shapes, window_counts
