import mlx.core as mx
from mlx import nn


class Qwen3SmallTextAdapter(nn.Module):
    def __init__(
        self,
        in_dim: int = 6144,
        out_dim: int = 7680,
        hidden: int = 8192,
        n_slices: int = 6,
        attn_dim: int = 1024,
        attn_heads: int = 8,
        attn_blocks: int = 2,
        max_len: int = 256,
    ):
        super().__init__()
        self.max_len = max_len
        self.mlp = [
            _PerLayerNorm(in_dim, n_slices),
            nn.Linear(in_dim, hidden),
            nn.GELU(approx="tanh"),
            nn.Linear(hidden, hidden),
            nn.GELU(approx="tanh"),
            nn.Linear(hidden, out_dim),
        ]
        self.attn = _AttnMixer(in_dim, out_dim, attn_dim, attn_heads, attn_blocks, max_len)

    def __call__(self, x: mx.array) -> mx.array:
        x = x.astype(mx.float32)
        return self.apply_mlp(x) + self.attn(x)

    def apply_mlp(self, x: mx.array) -> mx.array:
        for layer in self.mlp:
            x = layer(x)
        return x


class _PerLayerNorm(nn.Module):
    def __init__(self, in_dim: int, n_slices: int):
        super().__init__()
        self.n_slices = n_slices
        self.slice_dim = in_dim // n_slices
        self.all = nn.LayerNorm(in_dim, eps=1e-5)

    def __call__(self, x: mx.array) -> mx.array:
        batch, length, _ = x.shape
        x = x.reshape(batch, length, self.n_slices, self.slice_dim)
        x = mx.fast.layer_norm(x, None, None, 1e-5)
        return self.all(x.reshape(batch, length, -1))


class _AttnBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.n1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = nn.MultiHeadAttention(dim, heads, bias=True)
        self.n2 = nn.LayerNorm(dim, eps=1e-5)
        self.ffn = [nn.Linear(dim, dim * 4), nn.GELU(approx="tanh"), nn.Linear(dim * 4, dim)]

    def __call__(self, x: mx.array) -> mx.array:
        h = self.n1(x)
        x = x + self.attn(h, h, h)
        h = self.n2(x)
        for layer in self.ffn:
            h = layer(h)
        return x + h


class _AttnMixer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dim: int, heads: int, blocks: int, max_len: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim, eps=1e-5)
        self.inp = nn.Linear(in_dim, dim)
        self.pos = mx.zeros((1, max_len, dim))
        self.blocks = [_AttnBlock(dim, heads) for _ in range(blocks)]
        self.out = nn.Linear(dim, out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.inp(self.norm(x)) + self.pos[:, : x.shape[1]]
        for block in self.blocks:
            h = block(h)
        return self.out(h)
