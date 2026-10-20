import mlx.core as mx
from mlx import nn


class Qwen21TextFusion(nn.Module):
    # The zen-image-edit text_fusion block: Qwen3.5-0.8B hidden-state slices (B, L, K*d) -> the 4096-dim
    # input of the Qwen-Image-2.1 txt_in. Output = MLP(x) + token attention(x) + slice mixer(x).
    def __init__(self, config: dict):
        super().__init__()
        if config.get("norm") != "ln-per-layer" or config.get("proj_layers", 0) < 2:
            raise ValueError(f"Unsupported text_fusion_config: {config}")
        in_dim, out_dim, hidden = config["in_dim"], config["out_dim"], config["hidden"]
        n_slices = config["n_slices"]
        self.mlp = [_PerLayerNorm(in_dim, n_slices), nn.Linear(in_dim, hidden), nn.GELU(approx="tanh")]
        for _ in range(config["proj_layers"] - 2):
            self.mlp += [nn.Linear(hidden, hidden), nn.GELU(approx="tanh")]
        self.mlp.append(nn.Linear(hidden, out_dim))
        self.attn = (
            _AttnMixer(
                in_dim, out_dim, config["attn_dim"], config["attn_heads"], config["attention"], config["max_len"]
            )
            if config.get("attention")
            else None
        )
        self.mixer = (
            _SliceMixer(
                n_slices, in_dim // n_slices, out_dim, config["mixer_heads"], config["mixer"], config["mixer_ffn"]
            )
            if config.get("mixer")
            else None
        )

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        # mask: (B, L), true for real tokens; the token attention ignores the padding.
        x = x.astype(mx.float32)
        out = x
        for layer in self.mlp:
            out = layer(out)
        if self.attn is not None:
            out = out + self.attn(x, mask)
        if self.mixer is not None:
            out = out + self.mixer(x)
        return out


class _PerLayerNorm(nn.Module):
    def __init__(self, in_dim: int, n_slices: int):
        super().__init__()
        self.n_slices = n_slices
        self.slice_dim = in_dim // n_slices
        self.all = nn.LayerNorm(in_dim, eps=1e-5)

    def __call__(self, x: mx.array) -> mx.array:
        batch, length, _ = x.shape
        x = mx.fast.layer_norm(x.reshape(batch, length, self.n_slices, self.slice_dim), None, None, 1e-5)
        return self.all(x.reshape(batch, length, -1))


class _AttnBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_mult: int):
        super().__init__()
        self.n1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = nn.MultiHeadAttention(dim, heads, bias=True)
        self.n2 = nn.LayerNorm(dim, eps=1e-5)
        self.ffn = [nn.Linear(dim, dim * ffn_mult), nn.GELU(approx="tanh"), nn.Linear(dim * ffn_mult, dim)]

    def __call__(self, x: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        h = self.n1(x)
        x = x + self.attn(h, h, h, mask=attention_mask)
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
        self.blocks = [_AttnBlock(dim, heads, 4) for _ in range(blocks)]
        self.out = nn.Linear(dim, out_dim)

    def __call__(self, x: mx.array, mask: mx.array | None) -> mx.array:
        h = self.inp(self.norm(x))
        length = h.shape[1]
        pos = self.pos[:, :length]
        if length > pos.shape[1]:
            # Positions past the trained table get no positional signal, as in the reference.
            pos = mx.pad(pos, [(0, 0), (0, length - pos.shape[1]), (0, 0)])
        h = h + pos
        attention_mask = None
        if mask is not None and not bool(mx.all(mask)):
            attention_mask = mx.where(mask.astype(mx.bool_)[:, None, None, :], 0.0, -mx.inf).astype(h.dtype)
        for block in self.blocks:
            h = block(h, attention_mask)
        return self.out(h)


class _SliceMixer(nn.Module):
    def __init__(self, n_slices: int, dim: int, out_dim: int, heads: int, blocks: int, ffn_mult: int):
        super().__init__()
        self.n_slices = n_slices
        self.dim = dim
        self.blocks = [_AttnBlock(dim, heads, ffn_mult) for _ in range(blocks)]
        self.proj = nn.Linear(n_slices, 1, bias=False)
        self.out = nn.Linear(dim, out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        batch, length, _ = x.shape
        h = mx.fast.layer_norm(x.reshape(batch, length, self.n_slices, self.dim), None, None, 1e-5)
        h = h.reshape(batch * length, self.n_slices, self.dim)
        for block in self.blocks:
            h = block(h)
        h = self.proj(h.swapaxes(1, 2)).squeeze(-1)
        return self.out(h).reshape(batch, length, -1)
