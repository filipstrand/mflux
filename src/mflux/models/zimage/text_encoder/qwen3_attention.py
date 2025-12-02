import mlx.core as mx
import mlx.nn as nn
from mlx.core.fast import scaled_dot_product_attention


class Qwen3Attention(nn.Module):
    """Qwen3 attention with Grouped Query Attention.

    Key differences from S3DiT attention:
    - Uses GQA: 32 query heads, 8 KV heads (4:1 ratio)
    - Different RoPE configuration (theta=1000000)
    - Has QK norms (q_norm, k_norm)
    """

    # Qwen3-4B architecture (from HF weights)
    HIDDEN_SIZE = 2560
    NUM_HEADS = 32  # Q heads
    NUM_KV_HEADS = 8  # KV heads (4:1 GQA ratio)
    HEAD_DIM = 128  # From q_norm.weight shape
    ROPE_THETA = 1000000.0
    NORM_EPS = 1e-6

    def __init__(self):
        super().__init__()

        # Projections
        self.q_proj = nn.Linear(self.HIDDEN_SIZE, self.NUM_HEADS * self.HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(self.HIDDEN_SIZE, self.NUM_KV_HEADS * self.HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(self.HIDDEN_SIZE, self.NUM_KV_HEADS * self.HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(self.NUM_HEADS * self.HEAD_DIM, self.HIDDEN_SIZE, bias=False)

        # QK norms (per head_dim)
        self.q_norm = nn.RMSNorm(self.HEAD_DIM, eps=self.NORM_EPS)
        self.k_norm = nn.RMSNorm(self.HEAD_DIM, eps=self.NORM_EPS)

        # RoPE cache for text positions
        self._rope_cache: dict[int, tuple[mx.array, mx.array]] = {}

    def _get_rope(self, seq_len: int) -> tuple[mx.array, mx.array]:
        """Get or compute RoPE for given sequence length."""
        if seq_len not in self._rope_cache:
            freqs = 1.0 / (self.ROPE_THETA ** (mx.arange(0, self.HEAD_DIM, 2) / self.HEAD_DIM))
            positions = mx.arange(seq_len)
            angles = positions[:, None] * freqs[None, :]
            cos = mx.cos(angles)
            sin = mx.sin(angles)
            self._rope_cache[seq_len] = (cos, sin)
        return self._rope_cache[seq_len]

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        """Forward pass with optional attention mask.

        Args:
            x: Input tensor [B, S, HIDDEN_SIZE]
            mask: Optional attention mask [S, S] or [B, 1, S, S]

        Returns:
            Output tensor [B, S, HIDDEN_SIZE]
        """
        B, S, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).reshape(B, S, self.NUM_HEADS, self.HEAD_DIM)
        k = self.k_proj(x).reshape(B, S, self.NUM_KV_HEADS, self.HEAD_DIM)
        v = self.v_proj(x).reshape(B, S, self.NUM_KV_HEADS, self.HEAD_DIM)

        # Apply QK norms (per head)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE
        cos, sin = self._get_rope(S)
        q, k = self._apply_rope(q, k, cos, sin)

        # Expand KV heads for GQA
        heads_per_kv = self.NUM_HEADS // self.NUM_KV_HEADS  # 4
        k = mx.repeat(k, heads_per_kv, axis=2)
        v = mx.repeat(v, heads_per_kv, axis=2)

        # Transpose: [B, heads, S, head_dim]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Scaled dot-product attention (fused kernel)
        scale = self.HEAD_DIM**-0.5
        out = scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

        # Reshape and project
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.o_proj(out)

    def _apply_rope(self, q: mx.array, k: mx.array, cos: mx.array, sin: mx.array) -> tuple[mx.array, mx.array]:
        """Apply rotary position embeddings.

        Args:
            q: Query tensor [B, S, heads, head_dim]
            k: Key tensor [B, S, kv_heads, head_dim]
            cos: Cosine frequencies [S, head_dim/2]
            sin: Sine frequencies [S, head_dim/2]

        Returns:
            Rotated Q and K tensors
        """
        q_embed = self._rotate(q, cos, sin)
        k_embed = self._rotate(k, cos, sin)
        return q_embed, k_embed

    def _rotate(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        """Rotate tensor with RoPE.

        Args:
            x: Input tensor [B, S, heads, head_dim]
            cos: Cosine frequencies [S, head_dim/2]
            sin: Sine frequencies [S, head_dim/2]

        Returns:
            Rotated tensor
        """
        # Split into halves
        x1 = x[..., : self.HEAD_DIM // 2]
        x2 = x[..., self.HEAD_DIM // 2 :]

        # Rotate: [-x2, x1]
        rotated = mx.concatenate([-x2, x1], axis=-1)

        # Broadcast cos/sin: [S, head_dim/2] -> [1, S, 1, head_dim/2]
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]

        # Duplicate cos/sin to match full head_dim
        cos = mx.concatenate([cos, cos], axis=-1)
        sin = mx.concatenate([sin, sin], axis=-1)

        return x * cos + rotated * sin
