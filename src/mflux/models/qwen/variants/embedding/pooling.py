"""Pooling strategies for embedding models.

Implements last-token pooling used by Qwen3-VL embedding models.
"""

import mlx.core as mx


def pool_last_token(
    hidden_states: mx.array,
    attention_mask: mx.array,
) -> mx.array:
    """Pool the last non-padding token from hidden states.

    This is the standard pooling strategy for Qwen3-VL embedding models.
    It extracts the hidden state at the last valid (non-padded) position
    for each sequence in the batch.

    Uses reverse indexing to match PyTorch's flip + argmax behavior,
    handling non-contiguous masks correctly (e.g., when image placeholders
    create gaps in the attention mask).

    Args:
        hidden_states: Hidden states [batch, seq_len, hidden_size]
        attention_mask: Attention mask [batch, seq_len] where 1 = valid token

    Returns:
        Pooled embeddings [batch, hidden_size]

    Example:
        >>> hidden = mx.ones((2, 10, 2048))  # batch=2, seq=10, hidden=2048
        >>> mask = mx.array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        ...                  [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])
        >>> pooled = pool_last_token(hidden, mask)
        >>> pooled.shape
        (2, 2048)
    """
    # Match PyTorch's _pooling_last exactly using reverse indexing:
    # MLX doesn't have flip, so we reverse the mask using [:, ::-1]
    seq_len = attention_mask.shape[1]
    flipped = attention_mask[:, ::-1]  # Reverse along sequence axis
    last_one_positions = mx.argmax(flipped, axis=1).astype(mx.int32)
    last_positions = seq_len - last_one_positions - 1

    # Gather the embeddings at the last positions
    batch_size = hidden_states.shape[0]
    batch_indices = mx.arange(batch_size)

    # Use advanced indexing to get last token embeddings
    pooled = hidden_states[batch_indices, last_positions]

    return pooled


def normalize_embeddings(
    embeddings: mx.array,
    eps: float = 1e-12,
) -> mx.array:
    """L2 normalize embeddings.

    Matches PyTorch's F.normalize behavior exactly for numerical parity.

    Args:
        embeddings: Embeddings [batch, hidden_size]
        eps: Small value to avoid division by zero (1e-12 matches PyTorch default)

    Returns:
        Normalized embeddings [batch, hidden_size]
    """
    norms = mx.sqrt(mx.sum(embeddings * embeddings, axis=-1, keepdims=True))
    # Use max(norms, eps) like PyTorch instead of norms + eps
    return embeddings / mx.maximum(norms, mx.array(eps))
