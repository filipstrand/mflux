import mlx.core as mx
from mlx import nn

from mflux.models.common.config.model_config import ModelConfig


class VAEBlockDecoder:
    # Runs a diffusers-layout VAE decoder (conv_in, mid block with attention, up blocks, conv_norm_out, conv_out)
    # one block at a time in NHWC, with an mx.eval after each block, so MLX frees the temporaries of a block
    # before the next block starts. The attention runs in query chunks.
    #
    # strips=True also casts the latent to bf16 and runs each resnet, upsample and output layer in row strips.
    # Each layer keeps its full bf16 output, but its fp32 GroupNorm copies and conv inputs exist for one strip
    # only. The GroupNorm statistics come from the full image, so the strips do not shift the colours (the
    # per-tile statistics of VAETiler do). strips=False keeps the input dtype and the GroupNorm of the modules.
    #
    # The decoder modules can be the layer itself (Flux2) or a wrapper with .conv / .norm (Z-Image).
    # resnet_norm_fp32 tells how the resnets of this VAE run GroupNorm when strips=False.
    LEGACY = "legacy"
    LOW_MEMORY = "low_memory"
    FULL_PRECISION = "full_precision"
    STRIP_ROWS = 128
    ATTENTION_CHUNK = 4096

    @staticmethod
    def decode(vae, latent: mx.array, strips: bool, resnet_norm_fp32: bool = True) -> mx.array:
        if strips:
            latent = latent.astype(ModelConfig.precision)
        x = mx.transpose((latent / vae.scaling_factor) + vae.shift_factor, (0, 2, 3, 1))
        if getattr(vae, "post_quant_conv", None) is not None:
            x = vae.post_quant_conv(x)
        decoder = vae.decoder
        # A conv_in module only adds the NCHW transposes around nn.Conv2d (Flux2) or wraps one in .conv (Z-Image).
        # This decode runs in NHWC, so it calls nn.Conv2d.__call__ directly. If a decoder changes conv_in in other
        # ways, change this line too.
        x = VAEBlockDecoder._evaluated(nn.Conv2d.__call__(VAEBlockDecoder._conv(decoder.conv_in), x))
        x = VAEBlockDecoder._resnet(decoder.mid_block.resnets[0], x, strips, resnet_norm_fp32)
        for attention in decoder.mid_block.attentions:
            x = VAEBlockDecoder._attention(attention, x)
        x = VAEBlockDecoder._resnet(decoder.mid_block.resnets[1], x, strips, resnet_norm_fp32)
        for up_block in decoder.up_blocks:
            for resnet in up_block.resnets:
                x = VAEBlockDecoder._resnet(resnet, x, strips, resnet_norm_fp32)
            for upsampler in up_block.upsamplers or []:
                x = VAEBlockDecoder._upsample(upsampler, x, strips)
        conv_out = VAEBlockDecoder._conv(decoder.conv_out)
        x = VAEBlockDecoder._map_rows(
            x,
            VAEBlockDecoder._norm(decoder.conv_norm_out),
            lambda source, r0, r1, height: VAEBlockDecoder._conv3x3_rows(conv_out, source, r0, r1, height),
            strips,
            norm_fp32=True,
        )
        return mx.transpose(x, (0, 3, 1, 2))

    @staticmethod
    def _resnet(block, x: mx.array, strips: bool, norm_fp32: bool) -> mx.array:
        h = VAEBlockDecoder._map_rows(
            x,
            block.norm1,
            lambda source, r0, r1, height: VAEBlockDecoder._conv3x3_rows(block.conv1, source, r0, r1, height),
            strips,
            norm_fp32,
        )

        def conv2_with_residual(source, r0: int, r1: int, height: int) -> mx.array:
            residual = x[:, r0:r1]
            if block.conv_shortcut is not None:
                residual = nn.Conv2d.__call__(block.conv_shortcut, residual)
            return VAEBlockDecoder._conv3x3_rows(block.conv2, source, r0, r1, height) + residual

        return VAEBlockDecoder._map_rows(h, block.norm2, conv2_with_residual, strips, norm_fp32)

    @staticmethod
    def _upsample(block, x: mx.array, strips: bool) -> mx.array:
        batch, low_height, low_width, channels = x.shape
        height = 2 * low_height

        def nearest_rows(a: int, b: int) -> mx.array:
            ia, ib = a // 2, (b + 1) // 2
            rows = mx.broadcast_to(x[:, ia:ib, None, :, None, :], (batch, ib - ia, 2, low_width, 2, channels))
            rows = rows.reshape(batch, 2 * (ib - ia), 2 * low_width, channels)
            return rows[:, a - 2 * ia : b - 2 * ia]

        step = VAEBlockDecoder.STRIP_ROWS if strips else height
        conv = block.conv
        parts = [
            VAEBlockDecoder._evaluated(
                VAEBlockDecoder._conv3x3_rows(conv, nearest_rows, r0, min(r0 + step, height), height)
            )
            for r0 in range(0, height, step)
        ]
        return VAEBlockDecoder._joined(parts)

    @staticmethod
    def _attention(block, x: mx.array) -> mx.array:
        # Chunked queries keep the score matrix at (chunk x tokens). At 2048² the full matrix is about 8.6 GB.
        batch, height, width, channels = x.shape
        normed = VAEBlockDecoder._module_norm(block.group_norm, x, norm_fp32=True)
        q, k, v = (
            mx.transpose(projection(normed).reshape(batch, height * width, 1, channels), (0, 2, 1, 3))
            for projection in (block.to_q, block.to_k, block.to_v)
        )
        mx.eval(q, k, v)
        scale = 1 / mx.sqrt(channels)
        parts = []
        for i in range(0, height * width, VAEBlockDecoder.ATTENTION_CHUNK):
            chunk = q[:, :, i : i + VAEBlockDecoder.ATTENTION_CHUNK]
            parts.append(VAEBlockDecoder._evaluated(mx.fast.scaled_dot_product_attention(chunk, k, v, scale=scale)))
        attended = mx.transpose(mx.concatenate(parts, axis=2), (0, 2, 1, 3)).reshape(batch, height, width, channels)
        to_out = block.to_out[0] if isinstance(block.to_out, list) else block.to_out
        return VAEBlockDecoder._evaluated(x + to_out(attended))

    @staticmethod
    def _map_rows(x: mx.array, norm: nn.GroupNorm, conv_rows, strips: bool, norm_fp32: bool) -> mx.array:
        # conv_rows(source, r0, r1, height) returns output rows [r0, r1). source(a, b) is norm + SiLU of rows [a, b).
        height = x.shape[1]
        if strips:
            mean, inv_std = VAEBlockDecoder._group_stats(x, norm)
            source = lambda a, b: VAEBlockDecoder._stats_norm_silu(x[:, a:b], norm, mean, inv_std)  # noqa: E731
            step = VAEBlockDecoder.STRIP_ROWS
        else:
            normed = nn.silu(VAEBlockDecoder._module_norm(norm, x, norm_fp32))
            source = lambda a, b: normed[:, a:b]  # noqa: E731
            step = height
        parts = [
            VAEBlockDecoder._evaluated(conv_rows(source, r0, min(r0 + step, height), height))
            for r0 in range(0, height, step)
        ]
        return VAEBlockDecoder._joined(parts)

    @staticmethod
    def _conv3x3_rows(conv: nn.Conv2d, source, r0: int, r1: int, height: int) -> mx.array:
        # One halo row above and below the strip. Zero padding exists only at the image edges.
        VAEBlockDecoder._check_conv3x3(conv)
        a, b = max(r0 - 1, 0), min(r1 + 1, height)
        rows = mx.pad(source(a, b), ((0, 0), (1 if r0 == 0 else 0, 1 if r1 == height else 0), (1, 1), (0, 0)))
        return mx.conv2d(rows, conv.weight) + conv.bias

    @staticmethod
    def _check_conv3x3(conv: nn.Conv2d) -> None:
        # The halo rows and the padding in _conv3x3_rows are correct only for this conv shape.
        # nn.Conv2d keeps an int dilation as given, so 1 and (1, 1) are both correct.
        shape = (conv.weight.shape[1:3], conv.stride, conv.padding, conv.groups)
        if shape != ((3, 3), (1, 1), (1, 1), 1) or conv.dilation not in (1, (1, 1)):
            raise ValueError("VAEBlockDecoder needs 3x3 convs with stride 1, padding 1, dilation 1 and groups 1.")

    @staticmethod
    def _group_stats(x: mx.array, norm: nn.GroupNorm) -> tuple[mx.array, mx.array]:
        # The strip statistics use the PyTorch grouping: consecutive channels form one group.
        if not norm.pytorch_compatible:
            raise ValueError("VAEBlockDecoder strips need GroupNorm with pytorch_compatible=True.")
        batch, height, _, channels = x.shape
        groups = norm.num_groups
        count = x.size // (batch * groups)

        def strips():
            for r0 in range(0, height, VAEBlockDecoder.STRIP_ROWS):
                rows = x[:, r0 : r0 + VAEBlockDecoder.STRIP_ROWS].astype(mx.float32)
                yield rows.reshape(batch, -1, groups, channels // groups)

        total = mx.zeros((batch, groups), dtype=mx.float32)
        for rows in strips():
            total = VAEBlockDecoder._evaluated(total + rows.sum(axis=(1, 3)))
        mean = total / count
        squares = mx.zeros((batch, groups), dtype=mx.float32)
        for rows in strips():
            diff = rows - mean[:, None, :, None]
            squares = VAEBlockDecoder._evaluated(squares + (diff * diff).sum(axis=(1, 3)))
        return mean, mx.rsqrt(squares / count + norm.eps)

    @staticmethod
    def _stats_norm_silu(rows: mx.array, norm: nn.GroupNorm, mean: mx.array, inv_std: mx.array) -> mx.array:
        batch, height, width, channels = rows.shape
        groups = norm.num_groups
        x = rows.astype(mx.float32).reshape(batch, height, width, groups, channels // groups)
        x = (x - mean[:, None, None, :, None]) * inv_std[:, None, None, :, None]
        x = x.reshape(batch, height, width, channels) * norm.weight + norm.bias
        return nn.silu(x.astype(ModelConfig.precision))

    @staticmethod
    def _module_norm(norm: nn.GroupNorm, x: mx.array, norm_fp32: bool) -> mx.array:
        if not norm_fp32:
            return nn.GroupNorm.__call__(norm, x)
        return nn.GroupNorm.__call__(norm, x.astype(mx.float32)).astype(ModelConfig.precision)

    @staticmethod
    def _conv(module) -> nn.Conv2d:
        return module if isinstance(module, nn.Conv2d) else module.conv

    @staticmethod
    def _norm(module) -> nn.GroupNorm:
        return module if isinstance(module, nn.GroupNorm) else module.norm

    @staticmethod
    def _joined(parts: list[mx.array]) -> mx.array:
        return parts[0] if len(parts) == 1 else VAEBlockDecoder._evaluated(mx.concatenate(parts, axis=1))

    @staticmethod
    def _evaluated(x: mx.array) -> mx.array:
        mx.eval(x)
        return x
