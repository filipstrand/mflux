"""Jina CLIP v2 text encoder for NewBie-image.

Jina CLIP v2 provides semantic alignment:
- 1024 hidden dimension
- 24 transformer layers
- Standard multi-head attention
- GELU activation
"""

import mlx.core as mx
import mlx.nn as nn


class JinaCLIPAttention(nn.Module):
    """Multi-head attention for Jina CLIP."""

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_heads: int = 16,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.scale = self.head_dim ** -0.5

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
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose: [batch, heads, seq, head_dim]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Attention
        attn_weights = mx.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = mx.softmax(attn_weights, axis=-1)
        attn_output = mx.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        output = self.out_proj(attn_output)

        return output


class JinaCLIPMLP(nn.Module):
    """MLP for Jina CLIP with GELU activation."""

    def __init__(
        self,
        hidden_dim: int = 1024,
        intermediate_dim: int = 4096,
    ):
        super().__init__()

        self.fc1 = nn.Linear(hidden_dim, intermediate_dim, bias=True)
        self.fc2 = nn.Linear(intermediate_dim, hidden_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class JinaCLIPEncoderLayer(nn.Module):
    """Single transformer layer for Jina CLIP."""

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_heads: int = 16,
        intermediate_dim: int = 4096,
    ):
        super().__init__()

        self.self_attn = JinaCLIPAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
        )
        self.mlp = JinaCLIPMLP(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
        )

        self.layer_norm1 = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.layer_norm2 = nn.LayerNorm(hidden_dim, eps=1e-5)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        # Self-attention with residual (post-norm style)
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class JinaCLIPEncoder(nn.Module):
    """Jina CLIP v2 text encoder.

    Args:
        vocab_size: Vocabulary size (30522 for BERT-style tokenizer)
        hidden_dim: Hidden dimension (1024)
        num_layers: Number of transformer layers (24)
        num_heads: Number of attention heads (16)
        intermediate_dim: MLP intermediate dimension (4096)
        max_position_embeddings: Maximum sequence length (512)
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_dim: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        intermediate_dim: int = 4096,
        max_position_embeddings: int = 512,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Token and position embeddings
        self.word_embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_dim)

        # Layer norm after embeddings
        self.embeddings_layer_norm = nn.LayerNorm(hidden_dim, eps=1e-5)

        # Transformer layers
        self.layers = [
            JinaCLIPEncoderLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
            )
            for _ in range(num_layers)
        ]

        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(hidden_dim, eps=1e-5)

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
        batch_size, seq_len = input_ids.shape

        # Get position IDs
        position_ids = mx.arange(seq_len)[None, :]  # [1, seq_len]

        # Embed tokens and positions
        word_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)

        hidden_states = word_embeds + position_embeds
        hidden_states = self.embeddings_layer_norm(hidden_states)

        # Create attention mask bias
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -1e9

        # Process through layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        # Final norm
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states

    def get_pooled_output(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        """
        Get pooled output (CLS token representation).

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Pooled output [batch, hidden_dim]
        """
        hidden_states = self.__call__(input_ids, attention_mask)
        # Return CLS token (first token)
        return hidden_states[:, 0, :]
