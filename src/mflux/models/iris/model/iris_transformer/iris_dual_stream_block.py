import mlx.core as mx
from mlx import nn

from mflux.models.iris.model.iris_transformer.iris_feed_forward import IrisSwiGLU
from mflux.models.iris.model.iris_transformer.iris_joint_attention import IrisJointAttention
from mflux.models.iris.model.iris_transformer.iris_modulation_bias import IrisModulationBias
from mflux.models.iris.model.iris_transformer.iris_rms_norm import IrisRMSNorm


class IrisDualStreamBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_ratio: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.norm_x1 = IrisRMSNorm(dim, eps)
        self.norm_x2 = IrisRMSNorm(dim, eps)
        self.norm_y1 = IrisRMSNorm(dim, eps)
        self.norm_y2 = IrisRMSNorm(dim, eps)
        self.attn = IrisJointAttention(dim, num_heads, num_kv_heads, eps)
        self.attn_gate_x = nn.Linear(dim, dim, bias=False)
        self.attn_gate_y = nn.Linear(dim, dim, bias=False)
        self.attn_post_norm_x = IrisRMSNorm(dim, eps)
        self.attn_post_norm_y = IrisRMSNorm(dim, eps)
        self.mlp_x = IrisSwiGLU(dim, mlp_ratio)
        self.mlp_y = IrisSwiGLU(dim, mlp_ratio)
        self.mlp_post_norm_x = IrisRMSNorm(dim, eps)
        self.mlp_post_norm_y = IrisRMSNorm(dim, eps)
        self.adaln_img = IrisModulationBias(dim)
        self.adaln_txt = IrisModulationBias(dim)

    def __call__(
        self,
        x: mx.array,
        y: mx.array,
        img_core: mx.array,
        txt_core: mx.array,
        rope_img: tuple[mx.array, mx.array],
        rope_txt: tuple[mx.array, mx.array],
    ) -> tuple[mx.array, mx.array]:
        xs1, xc1, xg1, xs2, xc2, xg2 = mx.split(self.adaln_img(img_core), 6, axis=-1)
        ys1, yc1, yg1, ys2, yc2, yg2 = mx.split(self.adaln_txt(txt_core), 6, axis=-1)
        hx = self.norm_x1(x) * (1 + xc1) + xs1
        hy = self.norm_y1(y) * (1 + yc1) + ys1
        attn_x, attn_y = self.attn(
            hx,
            hy,
            rope_img,
            rope_txt,
            gate_x=mx.sigmoid(self.attn_gate_x(hx)),
            gate_y=mx.sigmoid(self.attn_gate_y(hy)),
        )
        x = x + xg1 * self.attn_post_norm_x(attn_x)
        x = x + xg2 * self.mlp_post_norm_x(self.mlp_x(self.norm_x2(x) * (1 + xc2) + xs2))
        y = y + yg1 * self.attn_post_norm_y(attn_y)
        y = y + yg2 * self.mlp_post_norm_y(self.mlp_y(self.norm_y2(y) * (1 + yc2) + ys2))
        return x, y
