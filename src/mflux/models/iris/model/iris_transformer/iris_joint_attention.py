import mlx.core as mx
from mlx import nn

from mflux.models.iris.model.iris_transformer.iris_rms_norm import IrisRMSNorm
from mflux.models.iris.model.iris_transformer.iris_rope import IrisRope


class IrisJointAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.q_proj_x = nn.Linear(dim, dim, bias=False)
        self.k_proj_x = nn.Linear(dim, kv_dim, bias=False)
        self.v_proj_x = nn.Linear(dim, kv_dim, bias=False)
        self.q_proj_y = nn.Linear(dim, dim, bias=False)
        self.k_proj_y = nn.Linear(dim, kv_dim, bias=False)
        self.v_proj_y = nn.Linear(dim, kv_dim, bias=False)
        self.q_norm_x = IrisRMSNorm(self.head_dim, eps)
        self.k_norm_x = IrisRMSNorm(self.head_dim, eps)
        self.q_norm_y = IrisRMSNorm(self.head_dim, eps)
        self.k_norm_y = IrisRMSNorm(self.head_dim, eps)
        self.proj_x = nn.Linear(dim, dim, bias=True)
        self.proj_y = nn.Linear(dim, dim, bias=True)

    def __call__(
        self,
        x: mx.array,
        y: mx.array,
        rope_img: tuple[mx.array, mx.array],
        rope_txt: tuple[mx.array, mx.array],
        gate_x: mx.array,
        gate_y: mx.array,
    ) -> tuple[mx.array, mx.array]:
        # q and k stay float32 from the norm to the softmax: bf16 rounding there perturbs the attention logits.
        batch, n_img, dim = x.shape
        n_txt = y.shape[1]
        qx = self.q_proj_x(x).astype(mx.float32).reshape(batch, n_img, self.num_heads, self.head_dim)
        kx = self.k_proj_x(x).astype(mx.float32).reshape(batch, n_img, self.num_kv_heads, self.head_dim)
        vx = self.v_proj_x(x).reshape(batch, n_img, self.num_kv_heads, self.head_dim)
        qy = self.q_proj_y(y).astype(mx.float32).reshape(batch, n_txt, self.num_heads, self.head_dim)
        ky = self.k_proj_y(y).astype(mx.float32).reshape(batch, n_txt, self.num_kv_heads, self.head_dim)
        vy = self.v_proj_y(y).reshape(batch, n_txt, self.num_kv_heads, self.head_dim)
        qx, kx = IrisRope.apply(self.q_norm_x(qx), rope_img), IrisRope.apply(self.k_norm_x(kx), rope_img)
        qy, ky = IrisRope.apply(self.q_norm_y(qy), rope_txt), IrisRope.apply(self.k_norm_y(ky), rope_txt)
        q = mx.concatenate([qy, qx], axis=1).transpose(0, 2, 1, 3)
        k = mx.concatenate([ky, kx], axis=1).transpose(0, 2, 1, 3)
        v = mx.concatenate([vy, vx], axis=1).transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(q, k, v.astype(mx.float32), scale=self.head_dim**-0.5)
        out = out.astype(x.dtype)
        out = out.transpose(0, 2, 1, 3).reshape(batch, n_txt + n_img, dim)
        return self.proj_x(out[:, n_txt:] * gate_x), self.proj_y(out[:, :n_txt] * gate_y)
