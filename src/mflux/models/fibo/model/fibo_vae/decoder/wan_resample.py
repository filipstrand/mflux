import mlx.core as mx
from mlx import nn

from mflux.models.fibo.model.fibo_vae.decoder.wan_causal_conv_3d import WanCausalConv3d


class WanResample(nn.Module):
    def __init__(self, dim: int, mode: str, upsample_out_dim: int = None):
        super().__init__()
        self.dim = dim
        self.mode = mode

        if upsample_out_dim is None:
            upsample_out_dim = dim // 2

        if mode == "upsample3d":
            self.time_conv = WanCausalConv3d(dim, dim * 2, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
            self.resample_conv = nn.Conv2d(dim, upsample_out_dim, kernel_size=3, stride=1, padding=1)
        elif mode == "upsample2d":
            self.resample_conv = nn.Conv2d(dim, upsample_out_dim, kernel_size=3, stride=1, padding=1)
            self.time_conv = None
        elif mode == "downsample2d":
            # Matches ZeroPad2d((0,1,0,1)) + Conv2d(dim, dim, 3, stride=2, padding=0)
            self.resample_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=0)
            self.time_conv = None
        elif mode == "downsample3d":
            # For feature_cache=None, PyTorch also only applies the spatial downsample here.
            # We mirror the same conv; the temporal time_conv path is only used with caching.
            self.resample_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=0)
            self.time_conv = None
        else:
            raise ValueError(f"Unsupported resample mode: {mode}")

    def __call__(self, x: mx.array, block_idx: int | None = None) -> mx.array:
        b, c, t, h, w = x.shape
        if self.mode in ("upsample2d", "upsample3d"):
            if self.mode == "upsample3d" and self.time_conv is not None:
                x = self.time_conv(x)
                x = mx.reshape(x, (b, 2, c, t, h, w))
                x = mx.transpose(x, (0, 2, 3, 1, 4, 5))
                x = mx.reshape(x, (b, c, t * 2, h, w))
                t = t * 2
            x = mx.transpose(x, (0, 2, 1, 3, 4))
            x = mx.reshape(x, (b * t, c, h, w))
            x = mx.transpose(x, (0, 2, 3, 1))
            x = mx.repeat(x, 2, axis=1)
            x = mx.repeat(x, 2, axis=2)
            x = self.resample_conv(x)
            x = mx.transpose(x, (0, 3, 1, 2))
            new_c = x.shape[1]
            new_h, new_w = x.shape[2], x.shape[3]
            x = mx.reshape(x, (b, t, new_c, new_h, new_w))
            x = mx.transpose(x, (0, 2, 1, 3, 4))
            return x

        # downsample modes
        x = mx.transpose(x, (0, 2, 1, 3, 4))
        x = mx.reshape(x, (b * t, c, h, w))
        x = mx.transpose(x, (0, 2, 3, 1))
        # ZeroPad2d((0,1,0,1)): pad right=1, bottom=1
        x = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])
        x = self.resample_conv(x)
        x = mx.transpose(x, (0, 3, 1, 2))
        new_c = x.shape[1]
        new_h, new_w = x.shape[2], x.shape[3]
        x = mx.reshape(x, (b, t, new_c, new_h, new_w))
        x = mx.transpose(x, (0, 2, 1, 3, 4))
        return x
