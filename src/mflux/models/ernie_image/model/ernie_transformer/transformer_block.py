import mlx.core as mx
from mlx import nn

from mflux.models.ernie_image.model.ernie_transformer.attention import ErnieAttention
from mflux.models.ernie_image.model.ernie_transformer.feed_forward import ErnieFeedForward


class ErnieTransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, ffn_hidden_size: int, eps: float, qk_layernorm: bool):
        super().__init__()
        self.adaLN_sa_ln = nn.RMSNorm(hidden_size, eps=eps)
        self.self_attention = ErnieAttention(hidden_size, num_heads, eps, qk_layernorm)
        self.adaLN_mlp_ln = nn.RMSNorm(hidden_size, eps=eps)
        self.mlp = ErnieFeedForward(hidden_size, ffn_hidden_size)

    def __call__(
        self,
        x: mx.array,
        freqs_cis: mx.array,
        temb: tuple,
        mask: mx.array | None,
    ) -> mx.array:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = temb

        # Self-attention with AdaLN pre-norm
        residual = x
        x = self.adaLN_sa_ln(x)
        x = (x.astype(mx.float32) * (1 + scale_msa.astype(mx.float32)) + shift_msa.astype(mx.float32)).astype(residual.dtype)
        x = residual + (gate_msa.astype(mx.float32) * self.self_attention(x, freqs_cis, mask).astype(mx.float32)).astype(residual.dtype)

        # FFN with AdaLN pre-norm
        residual = x
        x = self.adaLN_mlp_ln(x)
        x = (x.astype(mx.float32) * (1 + scale_mlp.astype(mx.float32)) + shift_mlp.astype(mx.float32)).astype(residual.dtype)
        x = residual + (gate_mlp.astype(mx.float32) * self.mlp(x).astype(mx.float32)).astype(residual.dtype)

        return x
