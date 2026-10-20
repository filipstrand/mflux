import mlx.core as mx
from mlx import nn

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.iris.model.iris_transformer.iris_dual_stream_block import IrisDualStreamBlock
from mflux.models.iris.model.iris_transformer.iris_embedders import (
    IrisPatchEmbedder,
    IrisPixelEmbedder,
    IrisTimestepEmbedder,
)
from mflux.models.iris.model.iris_transformer.iris_final_layer import IrisFinalLayer
from mflux.models.iris.model.iris_transformer.iris_pixel_block import IrisPixelBlock
from mflux.models.iris.model.iris_transformer.iris_rope import IrisRope
from mflux.models.iris.model.iris_transformer.iris_single_stream_block import IrisSingleStreamBlock
from mflux.models.iris.model.iris_transformer.iris_text_embedder import IrisTextEmbedder


class IrisTransformer(nn.Module):
    def __init__(
        self,
        hidden_size: int = 2560,
        depth: int = 24,
        dual_depth: int = 8,
        num_heads: int = 20,
        num_kv_heads: int = 5,
        patch_size: int = 16,
        in_channels: int = 3,
        text_len: int = 300,
        text_dim: int = 2560,
        text_num_layers: int = 12,
        text_layer_num_heads: int = 32,
        pixel_depth: int = 4,
        pixel_hidden_size: int = 16,
        pixel_attn_hidden_size: int = 1280,
        pixel_num_heads: int = 10,
        rope_theta: float = 10000.0,
        rope_scale: float = 16.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.text_len = text_len
        self.head_dim = hidden_size // num_heads
        self.pixel_head_dim = pixel_attn_hidden_size // pixel_num_heads
        self.rope_theta = rope_theta
        self.rope_scale = rope_scale

        self.s_embedder = IrisPatchEmbedder(patch_size * patch_size * in_channels, hidden_size)
        self.t_embedder = IrisTimestepEmbedder(hidden_size)
        self.y_embedder = IrisTextEmbedder(
            text_dim=text_dim,
            hidden_size=hidden_size,
            num_layers=text_num_layers,
            layer_num_heads=text_layer_num_heads,
            refiner_num_heads=num_heads,
            eps=eps,
        )
        self.y_pos_embedding = mx.zeros((1, text_len, hidden_size))
        self.modulation_cores = {
            "adaln_img": nn.Linear(hidden_size, 6 * hidden_size, bias=True),
            "adaln_txt": nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        }
        self.blocks = [
            IrisDualStreamBlock(hidden_size, num_heads, num_kv_heads, eps=eps)
            if i < dual_depth
            else IrisSingleStreamBlock(hidden_size, num_heads, num_kv_heads, eps=eps)
            for i in range(depth)
        ]
        self.pixel_embedder = IrisPixelEmbedder(in_channels, pixel_hidden_size, patch_size)
        self.pixel_blocks = [
            IrisPixelBlock(
                pixel_hidden_size,
                hidden_size,
                patch_size * patch_size,
                pixel_attn_hidden_size,
                pixel_num_heads,
                eps=eps,
            )
            for _ in range(pixel_depth)
        ]
        self.final_layer = IrisFinalLayer(pixel_hidden_size, in_channels, eps)

    def __call__(
        self,
        x: mx.array,
        t: mx.array,
        y: mx.array,
        y_mask: mx.array,
        captures: dict[str, mx.array] | None = None,
    ) -> mx.array:
        dtype = ModelConfig.precision
        p = self.patch_size
        batch, channels, height, width = x.shape
        grid = (height // p, width // p)
        n_patches = grid[0] * grid[1]
        x = x.astype(dtype)

        s = self.s_embedder(self._unfold(x))
        t_emb = self.t_embedder(t, dtype)
        cond = nn.silu(t_emb)
        y = self.y_embedder(y[:, : self.text_len].astype(dtype), y_mask[:, : self.text_len])
        self._capture(captures, "s_embedder", s)
        self._capture(captures, "t_embedder", t_emb)
        self._capture(captures, "y_embedder", y)
        y = y + self.y_pos_embedding[:, : y.shape[1]].astype(y.dtype)

        rope_img = IrisRope.rope_2d(self.head_dim, *grid, self.rope_theta, self.rope_scale)
        rope_txt = IrisRope.rope_1d(self.head_dim, y.shape[1], self.rope_theta)
        img_core = self.modulation_cores["adaln_img"](cond)
        txt_core = self.modulation_cores["adaln_txt"](cond)

        for i, block in enumerate(self.blocks):
            if isinstance(block, IrisDualStreamBlock):
                s, y = block(s, y, img_core, txt_core, rope_img, rope_txt)
            else:
                s, y = block(s, y, img_core, rope_img, rope_txt)
            self._capture(captures, f"blocks.{i}.s", s)
            self._capture(captures, f"blocks.{i}.y", y)

        s = nn.silu(t_emb + s)
        s_cond = s.reshape(batch * n_patches, -1)
        pixels = self.pixel_embedder(x)
        self._capture(captures, "pixel_embedder", pixels)
        rope_pix = IrisRope.rope_2d(self.pixel_head_dim, *grid, self.rope_theta, self.rope_scale)
        for i, block in enumerate(self.pixel_blocks):
            pixels = block(pixels, s_cond, rope_pix, grid)
            self._capture(captures, f"pixel_blocks.{i}", pixels)
        return self._fold(self.final_layer(pixels), batch, grid)

    def _unfold(self, x: mx.array) -> mx.array:
        p = self.patch_size
        batch, channels, height, width = x.shape
        patches = x.reshape(batch, channels, height // p, p, width // p, p).transpose(0, 2, 4, 1, 3, 5)
        return patches.reshape(batch, (height // p) * (width // p), channels * p * p)

    def _fold(self, pixels: mx.array, batch: int, grid: tuple[int, int]) -> mx.array:
        p = self.patch_size
        out = pixels.reshape(batch, grid[0], grid[1], p, p, self.in_channels).transpose(0, 5, 1, 3, 2, 4)
        return out.reshape(batch, self.in_channels, grid[0] * p, grid[1] * p)

    @staticmethod
    def _capture(captures: dict[str, mx.array] | None, name: str, value: mx.array) -> None:
        if captures is not None:
            captures[name] = value
