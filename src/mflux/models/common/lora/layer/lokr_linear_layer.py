import mlx.core as mx
from mlx import nn


def reconstruct_lokr_delta(
    *,
    alpha: float,
    rank: int,
    base_shape: tuple[int, int],
    w1: mx.array | None = None,
    w1_a: mx.array | None = None,
    w1_b: mx.array | None = None,
    w2: mx.array | None = None,
    w2_a: mx.array | None = None,
    w2_b: mx.array | None = None,
) -> mx.array:
    """Reconstruct a LoKr weight delta ``ΔW`` for a Linear layer, matching PEFT's
    ``LoKrLayer.get_delta_weight``:

        ``ΔW = (alpha / rank) · kron(w1, w2)``  reshaped to the base weight ``[out, in]``.

    Each Kronecker factor is provided either full (``w1`` / ``w2``) or as a low-rank
    product (``w1_a @ w1_b`` / ``w2_a @ w2_b``) — PEFT decomposes the second factor
    (and optionally the first) when the rank is below the factored dimension. The
    conv-only CP form (``lokr_t2``) is not produced for Linear targets and is
    unsupported here. ``rank_dropout`` is training-only, so inference ignores it.

    The result has the base weight's logical shape ``[out_dims, in_dims]`` (the
    reshape mirrors PEFT and is a no-op when the Kronecker factorization already
    lines up, which it does for LoKr-trained Linear adapters).
    """
    factor1 = w1 if w1 is not None else mx.matmul(w1_a, w1_b)
    factor2 = w2 if w2 is not None else mx.matmul(w2_a, w2_b)
    delta = mx.kron(factor1, factor2) * (float(alpha) / float(rank))
    return delta.reshape(base_shape)


class LoKrLinear(nn.Module):
    """A base Linear plus a fused LoKr delta, applied as a forward-time residual
    exactly like :class:`LoRALinear` (``base_out + scale · x · ΔWᵀ``) so it composes
    with the same quantized/non-quantized base layers and the same per-LoRA ``scale``
    knob (``scale = 0`` is a bit-exact no-op).

    Unlike LoRA, LoKr cannot be expressed as a low-rank ``A·B`` residual, so the full
    Kronecker delta ``ΔW`` (shape ``[out, in]``) is reconstructed once at construction
    (see :func:`reconstruct_lokr_delta`) and reused every step.
    """

    @staticmethod
    def from_linear(
        linear: nn.Linear | nn.QuantizedLinear,
        *,
        delta: mx.array,
        scale: float = 1.0,
    ) -> "LoKrLinear":
        layer = LoKrLinear(delta=delta, scale=scale)
        layer.linear = linear
        return layer

    @staticmethod
    def base_weight_shape(linear: nn.Linear | nn.QuantizedLinear) -> tuple[int, int]:
        """Logical ``[out_dims, in_dims]`` of the base layer, accounting for the
        packed weight of a quantized linear (mirrors ``LoRALinear.from_linear``)."""
        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims *= 32 // linear.bits
        return output_dims, input_dims

    def __init__(self, delta: mx.array, scale: float = 1.0):
        super().__init__()
        self.delta = delta
        self.scale = scale

    def __call__(self, x):
        base_out = self.linear(x)
        # The reconstructed delta is held at the factors' precision (typically fp32);
        # cast to the activation dtype so it composes with a bf16/quantized base.
        lokr_out = mx.matmul(x, self.delta.astype(x.dtype).T)
        return base_out + self.scale * lokr_out
