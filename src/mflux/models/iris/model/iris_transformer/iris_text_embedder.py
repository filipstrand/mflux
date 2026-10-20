import mlx.core as mx
from mlx import nn

from mflux.models.iris.model.iris_transformer.iris_feed_forward import IrisMLP, IrisSwiGLU
from mflux.models.iris.model.iris_transformer.iris_rms_norm import IrisRMSNorm
from mflux.models.iris.model.iris_transformer.iris_self_attention import IrisSelfAttention


class IrisLayerwiseBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, eps: float = 1e-6):
        super().__init__()
        self.norm1 = IrisRMSNorm(dim, eps)
        self.attn = IrisSelfAttention(dim, num_heads, qk_norm=False, eps=eps)
        self.norm2 = IrisRMSNorm(dim, eps)
        self.mlp = IrisMLP(dim, int(dim * mlp_ratio), activation="silu")

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class IrisTextAdapterBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, eps: float = 1e-6):
        super().__init__()
        self.norm1 = IrisRMSNorm(dim, eps)
        self.attn = IrisSelfAttention(dim, num_heads, qk_norm=True, eps=eps)
        self.norm2 = IrisRMSNorm(dim, eps)
        self.mlp = IrisSwiGLU(dim, mlp_ratio)

    def __call__(self, y: mx.array, mask: mx.array) -> mx.array:
        y = y + self.attn(self.norm1(y), mask=mask)
        return y + self.mlp(self.norm2(y))


class IrisTextRefiner(nn.Module):
    def __init__(self, text_dim: int, hidden_size: int, num_blocks: int, num_heads: int, mlp_ratio: float, eps: float):
        super().__init__()
        self.proj = nn.Linear(text_dim, hidden_size, bias=True)
        self.blocks = [IrisTextAdapterBlock(hidden_size, num_heads, mlp_ratio, eps) for _ in range(num_blocks)]
        self.norm = IrisRMSNorm(hidden_size, eps)

    def __call__(self, y: mx.array, mask: mx.array) -> mx.array:
        keep = mask.astype(mx.bool_)[:, None, None, :]
        attn_mask = keep | mx.eye(y.shape[1], dtype=mx.bool_)
        y = self.proj(y)
        for block in self.blocks:
            y = block(y, attn_mask)
        return self.norm(y)


class IrisTextEmbedder(nn.Module):
    def __init__(
        self,
        text_dim: int = 2560,
        hidden_size: int = 2560,
        num_layers: int = 12,
        layer_num_heads: int = 32,
        layer_mlp_ratio: float = 1.3,
        refiner_num_heads: int = 20,
        refiner_mlp_ratio: float = 4.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.layer_blocks = [IrisLayerwiseBlock(text_dim, layer_num_heads, layer_mlp_ratio, eps) for _ in range(2)]
        self.layer_pool = nn.Linear(num_layers, 1, bias=True)
        self.refiner = IrisTextRefiner(text_dim, hidden_size, 2, refiner_num_heads, refiner_mlp_ratio, eps)

    def __call__(self, y: mx.array, mask: mx.array) -> mx.array:
        batch, tokens, layers, dim = y.shape
        y = y.reshape(batch * tokens, layers, dim)
        for block in self.layer_blocks:
            y = block(y)
        y = self.layer_pool(y.transpose(0, 2, 1)).squeeze(-1)
        return self.refiner(y.reshape(batch, tokens, dim), mask)
