"""Gemma3-4B-it text encoder for NewBie-image.

Gemma3 is the primary text encoder featuring:
- 2560 hidden dimension
- 36 transformer layers
- GQA attention (similar to NextDiT)
- RMSNorm normalization
- SwiGLU feed-forward
"""

import mlx.core as mx
import mlx.nn as nn


class Gemma3RMSNorm(nn.Module):
    """RMSNorm for Gemma3.

    Gemma3 uses a zero-centered RMSNorm with (1 + weight) scaling:
    - Weights initialized to zeros (not ones) for 1-centering
    - Formula: x * rsqrt(mean(x^2) + eps) * (1 + weight)
    - Epsilon: 1e-6
    """

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        # Initialize to zeros for 1-centering (Gemma3 specification)
        self.weight = mx.zeros((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # Gemma3 uses (1 + weight) scaling with zero-initialized weights
        # This provides better training stability and gradient flow
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


class Gemma3Attention(nn.Module):
    """Multi-head attention for Gemma3 with GQA support.

    Gemma3-4B uses:
    - 16 query heads
    - 8 KV heads (GQA)
    - 160 head dimension
    """

    def __init__(
        self,
        hidden_dim: int = 2560,
        num_heads: int = 16,
        num_kv_heads: int = 8,
        head_dim: int = 160,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_groups = num_heads // num_kv_heads

        # QKV projections
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)

        self.scale = head_dim ** -0.5

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Transpose: [batch, heads, seq, head_dim]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Expand KV for GQA
        k = mx.repeat(k, self.num_groups, axis=1)
        v = mx.repeat(v, self.num_groups, axis=1)

        # Attention
        attn_weights = mx.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = mx.softmax(attn_weights, axis=-1)
        attn_output = mx.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        output = self.o_proj(attn_output)

        return output


class Gemma3MLP(nn.Module):
    """SwiGLU MLP for Gemma3."""

    def __init__(
        self,
        hidden_dim: int = 2560,
        intermediate_dim: int = 13824,
    ):
        super().__init__()

        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.gelu(self.gate_proj(x)) * self.up_proj(x))


class Gemma3DecoderLayer(nn.Module):
    """Single transformer layer for Gemma3."""

    def __init__(
        self,
        hidden_dim: int = 2560,
        num_heads: int = 16,
        num_kv_heads: int = 8,
        head_dim: int = 160,
        intermediate_dim: int = 13824,
    ):
        super().__init__()

        self.self_attn = Gemma3Attention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        self.mlp = Gemma3MLP(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
        )

        self.input_layernorm = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.post_attention_layernorm = nn.RMSNorm(hidden_dim, eps=1e-6)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Gemma3Encoder(nn.Module):
    """Gemma3-4B-it text encoder for NewBie-image.

    Args:
        vocab_size: Vocabulary size (256000 for Gemma3)
        hidden_dim: Hidden dimension (2560)
        num_layers: Number of transformer layers (36)
        num_heads: Number of attention heads (16)
        num_kv_heads: Number of KV heads for GQA (8)
        head_dim: Per-head dimension (160)
        intermediate_dim: MLP intermediate dimension (13824)
    """

    def __init__(
        self,
        vocab_size: int = 256000,
        hidden_dim: int = 2560,
        num_layers: int = 36,
        num_heads: int = 16,
        num_kv_heads: int = 8,
        head_dim: int = 160,
        intermediate_dim: int = 13824,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Token embedding
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)

        # Transformer layers
        self.layers = [
            Gemma3DecoderLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                intermediate_dim=intermediate_dim,
            )
            for _ in range(num_layers)
        ]

        # Final layer norm
        self.norm = nn.RMSNorm(hidden_dim, eps=1e-6)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        """
        Encode input tokens.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Hidden states [batch, seq_len, hidden_dim]
        """
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)

        # Create causal mask if needed for encoder use
        # For text encoding, we typically use bidirectional attention
        if attention_mask is not None:
            # Convert to attention bias: [batch, 1, 1, seq_len]
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -1e9

        # Process through layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        # Final norm
        hidden_states = self.norm(hidden_states)

        return hidden_states
