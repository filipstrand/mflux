from typing import Optional

import mlx.core as mx
from mlx import nn


class IPAdapter(nn.Module):
    """
    IP-Adapter for image prompt conditioning.
    Allows using reference images as additional conditioning alongside text prompts.
    """

    def __init__(
        self,
        image_encoder_hidden_size: int = 1024,
        cross_attention_dim: int = 768,
        num_tokens: int = 4,
        scale: float = 1.0,
    ):
        super().__init__()

        self.image_encoder_hidden_size = image_encoder_hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.scale = scale

        # Image projection layer
        self.image_proj = nn.Linear(image_encoder_hidden_size, cross_attention_dim * num_tokens)

        # Layer norm for stability
        self.norm = nn.LayerNorm(cross_attention_dim)

    def __call__(self, image_embeds: mx.array) -> mx.array:
        """
        Project image embeddings to cross-attention space.

        Args:
            image_embeds: Image embeddings from CLIP or similar [batch_size, hidden_size]

        Returns:
            Projected embeddings [batch_size, num_tokens, cross_attention_dim]
        """
        batch_size = image_embeds.shape[0]

        # Project to token space
        image_prompt_embeds = self.image_proj(image_embeds)

        # Reshape to token format
        image_prompt_embeds = image_prompt_embeds.reshape(batch_size, self.num_tokens, self.cross_attention_dim)

        # Apply normalization
        image_prompt_embeds = self.norm(image_prompt_embeds)

        return image_prompt_embeds * self.scale


class MLPProjection(nn.Module):
    """
    MLP-based projection for IP-Adapter.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_tokens: int = 4,
    ):
        super().__init__()

        self.num_tokens = num_tokens
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim * num_tokens),
        )

    def __call__(self, x: mx.array) -> mx.array:
        batch_size = x.shape[0]
        x = self.mlp(x)
        return x.reshape(batch_size, self.num_tokens, -1)


class IPAdapterAttnProcessor:
    """
    Attention processor that incorporates IP-Adapter image conditioning.
    """

    def __init__(self, hidden_size: int, cross_attention_dim: int, scale: float = 1.0):
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale

        # Additional projection layers for image conditioning
        self.to_k_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)

    def __call__(
        self,
        attn: nn.Module,  # The attention module
        hidden_states: mx.array,
        encoder_hidden_states: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        image_embeds: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Apply attention with IP-Adapter conditioning.
        """
        batch_size, sequence_length, _ = hidden_states.shape

        # Standard attention computation
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Apply IP-Adapter conditioning if image embeddings are provided
        if image_embeds is not None:
            ip_key = self.to_k_ip(image_embeds)
            ip_value = self.to_v_ip(image_embeds)

            # Concatenate text and image keys/values
            key = mx.concatenate([key, ip_key], axis=1)
            value = mx.concatenate([value, ip_value], axis=1)

        # Reshape for multi-head attention
        query = query.reshape(batch_size, sequence_length, attn.heads, attn.head_dim)
        key = key.reshape(batch_size, -1, attn.heads, attn.head_dim)
        value = value.reshape(batch_size, -1, attn.heads, attn.head_dim)

        # Transpose for attention computation
        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)

        # Compute attention
        attention_scores = mx.matmul(query, key.transpose(0, 1, 3, 2)) / (attn.head_dim**0.5)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = mx.softmax(attention_scores, axis=-1)
        hidden_states = mx.matmul(attention_probs, value)

        # Reshape back
        hidden_states = hidden_states.transpose(0, 2, 1, 3)
        hidden_states = hidden_states.reshape(batch_size, sequence_length, attn.inner_dim)

        # Final projection
        hidden_states = attn.to_out[0](hidden_states)

        return hidden_states
