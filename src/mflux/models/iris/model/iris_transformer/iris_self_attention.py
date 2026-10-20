import mlx.core as mx
from mlx import nn

from mflux.models.iris.model.iris_transformer.iris_rms_norm import IrisRMSNorm
from mflux.models.iris.model.iris_transformer.iris_rope import IrisRope


class IrisSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_norm = IrisRMSNorm(self.head_dim, eps) if qk_norm else None
        self.k_norm = IrisRMSNorm(self.head_dim, eps) if qk_norm else None
        self.proj = nn.Linear(dim, dim, bias=True)

    def __call__(
        self,
        x: mx.array,
        rope: tuple[mx.array, mx.array] | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        batch, n, dim = x.shape
        q, k, v = mx.split(self.qkv(x).reshape(batch, n, 3 * self.num_heads, self.head_dim), 3, axis=2)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            q, k = IrisRope.apply(q, rope), IrisRope.apply(k, rope)
        q, k, v = (t.transpose(0, 2, 1, 3) for t in (q, k, v))
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5, mask=mask)
        return self.proj(out.transpose(0, 2, 1, 3).reshape(batch, n, dim))
