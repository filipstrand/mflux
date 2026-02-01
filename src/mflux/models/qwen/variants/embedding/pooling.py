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

    Note:
        Assumes attention masks have contiguous 1s followed by 0s (standard padding).
        For such masks, sum(mask) - 1 gives the last valid token position directly,
        avoiding the need for array flipping and improving performance.

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
    # For contiguous masks (1s followed by 0s), sum gives sequence length
    # Last valid position = sum - 1 (avoids array flipping overhead)
    last_positions = mx.sum(attention_mask, axis=1).astype(mx.int32) - 1

    # Gather the embeddings at the last positions
    batch_size = hidden_states.shape[0]
    batch_indices = mx.arange(batch_size)

    # Use advanced indexing to get last token embeddings
    pooled = hidden_states[batch_indices, last_positions]

    return pooled


def normalize_embeddings(
    embeddings: mx.array,
    eps: float = 1e-8,
) -> mx.array:
    """L2 normalize embeddings.

    Args:
        embeddings: Embeddings [batch, hidden_size]
        eps: Small value to avoid division by zero (1e-8 for float16 compatibility)

    Returns:
        Normalized embeddings [batch, hidden_size]
    """
    norms = mx.sqrt(mx.sum(embeddings * embeddings, axis=-1, keepdims=True))
    return embeddings / (norms + eps)
