from dataclasses import dataclass


@dataclass
class Krea2TransformerConfig:
    """Configuration for the Krea 2 single-stream MMDiT transformer (Turbo defaults).

    Mirrors diffusers' Krea2Transformer2DModel. The Krea 2 Turbo checkpoint stores these under
    abbreviated config keys (channels/features/heads/...); the loader maps them to these names.
    """

    in_channels: int = 64
    num_layers: int = 28
    attention_head_dim: int = 128
    num_attention_heads: int = 48
    num_key_value_heads: int = 12
    intermediate_size: int = 16384
    timestep_embed_dim: int = 256
    text_hidden_dim: int = 2560
    num_text_layers: int = 12
    text_num_attention_heads: int = 20
    text_num_key_value_heads: int = 20
    text_intermediate_size: int = 6912
    num_layerwise_text_blocks: int = 2
    num_refiner_text_blocks: int = 2
    axes_dims_rope: tuple[int, int, int] = (32, 48, 48)
    rope_theta: float = 1000.0
    norm_eps: float = 1e-5

    @property
    def hidden_size(self) -> int:
        return self.attention_head_dim * self.num_attention_heads
