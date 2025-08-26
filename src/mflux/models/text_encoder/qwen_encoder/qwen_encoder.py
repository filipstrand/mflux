import mlx.core as mx
from mlx import nn

from mflux.models.text_encoder.qwen_encoder.qwen_encoder_layer import QwenEncoderLayer
from mflux.models.text_encoder.qwen_encoder.qwen_rms_norm import QwenRMSNorm


class QwenEncoder(nn.Module):
    """
    Qwen2.5-VL Text Encoder implementation in MLX.

    This encoder produces text embeddings with shape [batch, seq_len, 3584]
    to match the joint_attention_dim used in the Qwen transformer.
    """

    def __init__(
        self,
        vocab_size: int = 152064,  # Actual Qwen2.5-VL vocab size
        hidden_size: int = 3584,  # Match joint_attention_dim
        num_hidden_layers: int = 28,  # Typical for Qwen2.5-7B
        num_attention_heads: int = 28,
        num_key_value_heads: int = 4,  # GQA
        intermediate_size: int = 18944,  # 3584 * 5.28 (typical ratio)
        max_position_embeddings: int = 128000,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,  # From config
        rope_scaling: dict = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        # Token embeddings
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # Transformer layers
        self.layers = [
            QwenEncoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                intermediate_size=intermediate_size,
                rms_norm_eps=rms_norm_eps,
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
            )
            for _ in range(num_hidden_layers)
        ]

        # Final layer norm
        self.norm = QwenRMSNorm(hidden_size, eps=rms_norm_eps)

    def __call__(
        self, input_ids: mx.array, attention_mask: mx.array | None = None, position_ids: mx.array | None = None
    ) -> mx.array:
        """
        Forward pass of the Qwen encoder.

        Args:
            input_ids: Token IDs of shape [batch_size, seq_len]
            attention_mask: Attention mask of shape [batch_size, seq_len]

        Returns:
            Hidden states of shape [batch_size, seq_len, hidden_size]
        """
        # Token embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_ids)

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        return hidden_states
