import mlx.core as mx
from mlx import nn

from mflux.models.iris.model.iris_transformer.iris_feed_forward import IrisSwiGLU
from mflux.models.iris.model.iris_transformer.iris_modulation_bias import IrisModulationBias
from mflux.models.iris.model.iris_transformer.iris_rms_norm import IrisRMSNorm
from mflux.models.iris.model.iris_transformer.iris_rope import IrisRope


class IrisSingleStreamBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_ratio: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.norm1 = IrisRMSNorm(dim, eps)
        self.norm2 = IrisRMSNorm(dim, eps)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.q_norm = IrisRMSNorm(self.head_dim, eps)
        self.k_norm = IrisRMSNorm(self.head_dim, eps)
        self.attn_gate = nn.Linear(dim, dim, bias=False)
        self.attn_proj = nn.Linear(dim, dim, bias=True)
        self.attn_post_norm = IrisRMSNorm(dim, eps)
        self.mlp = IrisSwiGLU(dim, mlp_ratio)
        self.mlp_post_norm = IrisRMSNorm(dim, eps)
        self.adaln = IrisModulationBias(dim)

    def __call__(
        self,
        x: mx.array,
        y: mx.array,
        img_core: mx.array,
        rope_img: tuple[mx.array, mx.array],
        rope_txt: tuple[mx.array, mx.array],
    ) -> tuple[mx.array, mx.array]:
        # q, k and v stay float32 from the projection to the softmax: bf16 rounding of q and k perturbs the logits.
        batch, n_img, dim = x.shape
        n_txt = y.shape[1]
        n_tokens = n_txt + n_img
        s1, c1, g1, s2, c2, g2 = mx.split(self.adaln(img_core), 6, axis=-1)
        tokens = mx.concatenate([y, x], axis=1)
        h = self.norm1(tokens) * (1 + c1) + s1
        q = self.q_norm(self.q_proj(h).astype(mx.float32).reshape(batch, n_tokens, self.num_heads, self.head_dim))
        k = self.k_norm(self.k_proj(h).astype(mx.float32).reshape(batch, n_tokens, self.num_kv_heads, self.head_dim))
        v = self.v_proj(h).astype(mx.float32).reshape(batch, n_tokens, self.num_kv_heads, self.head_dim)
        q = mx.concatenate([IrisRope.apply(q[:, :n_txt], rope_txt), IrisRope.apply(q[:, n_txt:], rope_img)], axis=1)
        k = mx.concatenate([IrisRope.apply(k[:, :n_txt], rope_txt), IrisRope.apply(k[:, n_txt:], rope_img)], axis=1)
        q, k, v = (t.transpose(0, 2, 1, 3) for t in (q, k, v))
        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5).astype(h.dtype)
        attn = attn.transpose(0, 2, 1, 3).reshape(batch, n_tokens, dim)
        attn = attn * mx.sigmoid(self.attn_gate(h))
        attn = self.attn_post_norm(self.attn_proj(attn))
        tokens = tokens + g1 * attn
        mlp = self.mlp(self.norm2(tokens) * (1 + c2) + s2)
        tokens = tokens + g2 * self.mlp_post_norm(mlp)
        return tokens[:, n_txt:], tokens[:, :n_txt]
