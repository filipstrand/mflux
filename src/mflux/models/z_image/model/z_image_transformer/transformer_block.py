import mlx.core as mx
from mlx import nn

from mflux.models.z_image.model.z_image_transformer.attention import ZImageAttention
from mflux.models.z_image.model.z_image_transformer.feed_forward import FeedForward


class ZImageTransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        norm_eps: float = 1e-5,
        qk_norm: bool = True,
        modulation: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.layer_id = layer_id
        self.modulation = modulation

        # Attention
        self.attention = ZImageAttention(
            dim=dim,
            n_heads=n_heads,
            qk_norm=qk_norm,
            eps=1e-5,
        )

        # Feed-forward (SwiGLU)
        hidden_dim = int(dim / 3 * 8)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=hidden_dim)

        # Normalization layers
        self.attention_norm1 = nn.RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = nn.RMSNorm(dim, eps=norm_eps)

        # AdaLN modulation
        if modulation:
            # Output: 4 * dim (scale_msa, gate_msa, scale_mlp, gate_mlp)
            self.adaLN_modulation = [nn.Linear(min(dim, 256), 4 * dim, bias=True)]

    def __call__(
        self,
        x: mx.array,
        attn_mask: mx.array,
        freqs_cis: mx.array,
        adaln_input: mx.array | None = None,
    ) -> mx.array:
        if self.modulation:
            assert adaln_input is not None

            # Get modulation parameters
            modulation = self.adaLN_modulation[0](adaln_input)
            modulation = mx.expand_dims(modulation, axis=1)  # (batch, 1, 4*dim)

            # Split into scale and gate for MSA and MLP
            scale_msa, gate_msa, scale_mlp, gate_mlp = mx.split(modulation, 4, axis=2)
            gate_msa = mx.tanh(gate_msa)
            gate_mlp = mx.tanh(gate_mlp)
            scale_msa = 1.0 + scale_msa
            scale_mlp = 1.0 + scale_mlp

            # Attention block with modulation
            attn_out = self.attention(
                self.attention_norm1(x) * scale_msa,
                attention_mask=attn_mask,
                freqs_cis=freqs_cis,
            )
            x = x + gate_msa * self.attention_norm2(attn_out)

            # FFN block with modulation
            ffn_out = self.feed_forward(self.ffn_norm1(x) * scale_mlp)
            x = x + gate_mlp * self.ffn_norm2(ffn_out)
        else:
            # Without modulation (for context_refiner)
            attn_out = self.attention(
                self.attention_norm1(x),
                attention_mask=attn_mask,
                freqs_cis=freqs_cis,
            )
            x = x + self.attention_norm2(attn_out)

            ffn_out = self.feed_forward(self.ffn_norm1(x))
            x = x + self.ffn_norm2(ffn_out)

        return x
