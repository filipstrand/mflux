import mlx.core as mx
import numpy as np
from mlx import nn

from mflux.models.moss_image.model.moss_image_transformer.moss_image_block import MOSSImageBlock
from mflux.models.moss_image.model.moss_image_transformer.moss_image_rope import MOSSImageRoPE
from mflux.models.moss_image.model.moss_image_transformer.moss_image_text_embedder import MOSSImageTextProjection


class MOSSImageTransformer(nn.Module):
    # Module names follow the official checkpoint (OpenMOSS-Team/MOSS-Image-SFT, transformer/),
    # so the weight mapping is the identity.
    def __init__(
        self,
        dim: int = 3072,
        ffn_dim: int = 8192,
        num_heads: int = 24,
        num_layers: int = 42,
        text_dim: int = 4096,
        freq_dim: int = 256,
        latent_channels: int = 128,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        if dim % num_heads or (dim // num_heads) % 8 or freq_dim % 2:
            raise ValueError("MOSS-Image needs dim divisible by num_heads, head_dim divisible by 8, and an even freq_dim.")  # fmt: off
        self.dim = dim
        self.freq_dim = freq_dim
        self.latent_channels = latent_channels
        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = MOSSImageTextProjection(text_dim, dim, ffn_dim, num_heads)
        self.rope = MOSSImageRoPE(dim // num_heads, rope_base)
        self.time_embedding = [nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)]
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = [MOSSImageBlock(dim, ffn_dim, num_heads) for _ in range(num_layers)]
        self.final_norm_video = nn.RMSNorm(dim, eps=1e-6)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)

    def __call__(
        self,
        hidden_states: mx.array,
        timestep: float,
        encoder_hidden_states: mx.array,
        skip_layers: frozenset[int] = frozenset(),
    ) -> mx.array:
        # hidden_states: (B, latent_channels, H, W); encoder_hidden_states: (B, S, text_dim)
        batch, channels, height, width = hidden_states.shape
        modulation, head = self._modulation(timestep, batch)

        video = hidden_states.reshape(batch, channels, height * width).transpose(0, 2, 1)
        video = self.video_embedder(video.astype(modulation.dtype))
        rope = self.rope(height, width, encoder_hidden_states.shape[1], dtype=video.dtype)
        text = self.text_embedder(encoder_hidden_states, rope[:, height * width :])

        x = mx.concatenate([video, text], axis=1)
        for index, layer in enumerate(self.layers):
            if index not in skip_layers:
                x = layer(x, rope, modulation)

        x = self.final_norm_video(x[:, : height * width])
        x = x * (1 + head[:, 1:2]) + head[:, 0:1]
        x = self.final_linear_video(x)
        return x.transpose(0, 2, 1).reshape(batch, channels, height, width)

    @staticmethod
    def check_skip_layers(layers, depth: int) -> frozenset[int]:
        layers = tuple(layers)
        if any(type(index) is not int or not 0 <= index < depth for index in layers):
            raise ValueError(f"uncond_skip_layers must hold layer indices in [0, {depth}), got {list(layers)}.")
        return frozenset(layers)

    def time_signal(self, sigmas: mx.array) -> mx.array:
        # The timestep embedding for each sigma. StepCache scores the steps with it.
        return self._time_embedding([float(sigma) * 1000.0 for sigma in sigmas.tolist()])

    def _modulation(self, timestep: float, batch: int) -> tuple[mx.array, mx.array]:
        time = self._time_embedding([float(timestep)])
        modulation = self.shared_block_modulation(time).reshape(1, 6, self.dim)
        head = self.video_head_modulation(time).reshape(1, 2, self.dim)
        return mx.broadcast_to(modulation, (batch, 6, self.dim)), mx.broadcast_to(head, (batch, 2, self.dim))

    def _time_embedding(self, timesteps: list[float]) -> mx.array:
        # The reference computes the sinusoid in float64. The timesteps are host scalars, so do that with numpy.
        half = self.freq_dim // 2
        phase = np.array(timesteps, dtype=np.float64)[:, None] * 10000.0 ** (-np.arange(half, dtype=np.float64) / half)
        embedding = mx.array(np.concatenate([np.cos(phase), np.sin(phase)], axis=-1).astype(np.float32))
        time = embedding.astype(self.shared_block_modulation.bias.dtype)
        for module in self.time_embedding:
            time = module(time)
        return time
