import mlx.core as mx
from mlx import nn

from mflux.models.common.compute_precision import ComputePrecision
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux.model.flux_transformer.ada_layer_norm_continuous import AdaLayerNormContinuous
from mflux.models.flux3.model.flux3_transformer.flux3_attention import Flux3Attention
from mflux.models.flux3.model.flux3_transformer.flux3_block import Flux3Block
from mflux.models.flux3.model.flux3_transformer.flux3_feed_forward import Flux3FeedForward
from mflux.models.flux3.model.flux3_transformer.flux3_modulation import Flux3Modulation
from mflux.models.flux3.model.flux3_transformer.flux3_pos_embed import Flux3PosEmbed
from mflux.models.flux3.model.flux3_transformer.flux3_timestep_embedder import Flux3TimestepEmbedder


class Flux3Transformer(nn.Module):
    IMAGE_MODES = ("image", "image_cond")
    MOD_MODES = ("image", "image_cond", "txt")

    def __init__(
        self,
        in_channels: int = 128,
        cond_in_channels: int = 128,
        num_layers: int = 5,
        num_single_layers: int = 28,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 20480,
        timestep_channels: int = 256,
        mlp_ratio: float = 3.0,
        axes_dims_rope: tuple[int, ...] = (32, 32, 32, 32),
        rope_theta: int = 10000,
        flow_map: bool = True,
    ):
        super().__init__()
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = Flux3PosEmbed(theta=rope_theta, axes_dim=axes_dims_rope)
        self.timestep_embedder = Flux3TimestepEmbedder(timestep_channels, self.inner_dim)
        # Only the distilled checkpoint is a flow map (its time embedding is the sum of the current and the target
        # time); the undistilled Base checkpoint has no target_timestep_embedder.
        self.flow_map = flow_map
        self.target_timestep_embedder = Flux3TimestepEmbedder(timestep_channels, self.inner_dim) if flow_map else None
        self.early_stream_modulations = {m: Flux3Modulation(self.inner_dim) for m in self.MOD_MODES}
        self.single_stream_modulations = {m: Flux3Modulation(self.inner_dim) for m in self.MOD_MODES}

        self.x_embedders = {
            "image": nn.Linear(in_channels, self.inner_dim, bias=False),
            "image_cond": nn.Linear(cond_in_channels, self.inner_dim, bias=False),
        }
        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim, bias=False)

        make_block = lambda: Flux3Block(  # noqa: E731
            dim=self.inner_dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            mlp_ratio=mlp_ratio,
        )
        self.mode_blocks = {m: [make_block() for _ in range(num_layers)] for m in self.IMAGE_MODES}
        self.txt_mode_blocks = [make_block() for _ in range(num_layers)]
        self.single_transformer_blocks = [make_block() for _ in range(num_single_layers)]

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, bias=False)
        self.proj_out = nn.Linear(self.inner_dim, in_channels, bias=False)

    def __call__(
        self,
        hidden_states: mx.array,
        img_ids: mx.array,
        encoder_hidden_states: mx.array,
        txt_ids: mx.array,
        timestep: mx.array,
        target_timestep: mx.array | None = None,
        cond_hidden_states: mx.array | None = None,
        cond_ids: mx.array | None = None,
    ) -> mx.array:
        dtype = ModelConfig.precision
        num_image = hidden_states.shape[1]
        has_cond = cond_hidden_states is not None and cond_hidden_states.shape[1] > 0

        # Image tokens see the current time, conditioning and text tokens see t = 0; a flow map adds the target time to every mode.
        if self.flow_map != (target_timestep is not None):
            raise ValueError("target_timestep is required by, and only by, a flow-map transformer")
        vec = {
            "image": self.timestep_embedder(timestep),
            "context": self.timestep_embedder(mx.zeros_like(timestep)),
        }
        if self.flow_map:
            target = self.target_timestep_embedder(target_timestep)
            vec = {k: v + target for k, v in vec.items()}
        vec_by_mode = {"image": vec["image"], "image_cond": vec["context"], "txt": vec["context"]}
        early = {m: self.early_stream_modulations[m](vec_by_mode[m]) for m in self.MOD_MODES}
        single = {m: self.single_stream_modulations[m](vec_by_mode[m]) for m in self.MOD_MODES}

        streams = {"image": self.x_embedders["image"](hidden_states.astype(dtype))}
        ids = {"image": img_ids}
        if has_cond:
            streams["image_cond"] = self.x_embedders["image_cond"](cond_hidden_states.astype(dtype))
            ids["image_cond"] = cond_ids
        streams["txt"] = self.context_embedder(encoder_hidden_states.astype(dtype))
        ids["txt"] = txt_ids
        rotary = {m: self.pos_embed(ids[m]) for m in streams}

        # Per-mode stage: each mode attends only within itself.
        for d in range(len(self.txt_mode_blocks)):
            for m in streams:
                block = self.txt_mode_blocks[d] if m == "txt" else self.mode_blocks[m][d]
                streams[m] = block(streams[m], rotary[m], early[m])

        # Single stream over [image, image_cond, text] with per-token modulation.
        order = list(streams)
        x = mx.concatenate([streams[m] for m in order], axis=1)
        single_rotary = self.pos_embed(mx.concatenate([ids[m] for m in order], axis=0))
        single_mod = tuple(
            mx.concatenate(
                [mx.broadcast_to(single[m][i], (x.shape[0], streams[m].shape[1], x.shape[-1])) for m in order], axis=1
            )  # fmt: off
            for i in range(3)
        )
        for block in self.single_transformer_blocks:
            x = block(x, single_rotary, single_mod)

        x = self.norm_out(x[:, :num_image], vec["image"])
        return self.proj_out(x)

    def apply_compute_precision(self, precision: ComputePrecision) -> None:
        precision.apply(self, (Flux3Attention, Flux3FeedForward))
