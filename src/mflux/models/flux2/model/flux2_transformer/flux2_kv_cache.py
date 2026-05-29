"""
KV-cache container for the FLUX.2-klein-9b-kv variant.

The 9B-KV checkpoint distilled from FLUX.2 [klein] 9B at 4 inference steps with
an attention-side optimisation: reference-image (and text) K/V tensors are
computed once on step 0 and re-used on steps 1-3, skipping redundant
reference-token processing. BFL's reference implementation lives in the
``Flux2KleinKVPipeline`` in upstream diffusers.

This container holds the per-layer K/V slots for both the double-stream
(``transformer_blocks``) and single-stream (``single_transformer_blocks``)
stacks, plus the slice metadata each attention layer needs to know.

mflux's concat order inside the transformer differs from diffusers'. In
mflux the post-concat sequence is ``[txt, target, ref]`` (because
``Flux2KleinEdit._predict`` concatenates ``[latents, image_latents]`` and then
``Flux2Attention`` prepends the text-derived stream). Diffusers uses
``[txt, ref, target]``. The cache protocol is identical; only the slice
indices differ — we slice ``[:, :, num_txt + num_target :, :]`` in extract
mode and splice the cached ref K/V at the *end* of the fresh K/V in cached
mode.
"""

from __future__ import annotations

from typing import Literal

import mlx.core as mx

CacheMode = Literal["extract", "cached"]
StreamType = Literal["double", "single"]


class Flux2KVCache:
    """Per-layer K/V slot store for FLUX.2-klein-9b-kv.

    Attributes
    ----------
    mode :
        ``"extract"`` populates the cache during the first denoise step,
        ``"cached"`` reads from it on subsequent steps.
    num_ref_tokens :
        Count of reference-image tokens (the static slice we cache).
    num_txt_tokens :
        Count of text-encoder tokens (also static across steps; needed in
        both modes so attention layers know where target tokens start).
    """

    def __init__(self, num_double_layers: int, num_single_layers: int) -> None:
        self._double: list[tuple[mx.array, mx.array] | None] = [None] * num_double_layers
        self._single: list[tuple[mx.array, mx.array] | None] = [None] * num_single_layers
        self.mode: CacheMode | None = None
        self.num_ref_tokens: int = 0
        self.num_txt_tokens: int = 0

    def configure(
        self,
        *,
        mode: CacheMode,
        num_ref_tokens: int,
        num_txt_tokens: int,
    ) -> None:
        self.mode = mode
        self.num_ref_tokens = int(num_ref_tokens)
        self.num_txt_tokens = int(num_txt_tokens)

    # ------- store (extract mode) ----------------------------------------

    def store(self, stream: StreamType, layer_idx: int, key: mx.array, value: mx.array) -> None:
        slot = (key, value)
        if stream == "double":
            self._double[layer_idx] = slot
        elif stream == "single":
            self._single[layer_idx] = slot
        else:
            raise ValueError(f"Unknown stream {stream!r}")

    # ------- load (cached mode) ------------------------------------------

    def load(self, stream: StreamType, layer_idx: int) -> tuple[mx.array, mx.array]:
        slot = self._double[layer_idx] if stream == "double" else self._single[layer_idx]
        if slot is None:
            raise RuntimeError(
                f"KV cache slot for {stream} layer {layer_idx} not populated; "
                "did you forget the extract pass on step 0?"
            )
        return slot

    # ------- inspection --------------------------------------------------

    def is_populated(self) -> bool:
        return all(s is not None for s in self._double) and all(s is not None for s in self._single)

    def reset(self) -> None:
        self._double = [None] * len(self._double)
        self._single = [None] * len(self._single)
        self.mode = None
        self.num_ref_tokens = 0
        self.num_txt_tokens = 0
