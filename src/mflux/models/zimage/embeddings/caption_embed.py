import mlx.core as mx
import mlx.nn as nn


class CaptionEmbed(nn.Module):
    """Project Qwen3 text embeddings to transformer dimension.

    RMSNorm followed by linear projection from text encoder hidden dim to S3-DiT dim.
    HF structure: cap_embedder.0 = RMSNorm(2560), cap_embedder.1 = Linear(2560, 3840)
    """

    CAP_FEAT_DIM = 2560  # Qwen3-4B hidden size
    HIDDEN_DIM = 3840  # S3-DiT dimension
    NORM_EPS = 1e-5

    def __init__(self):
        super().__init__()
        # Layer 0: RMSNorm (weight only, maps to cap_embedder.0.weight)
        self.linear1 = nn.RMSNorm(self.CAP_FEAT_DIM, eps=self.NORM_EPS)
        # Layer 1: Linear projection (maps to cap_embedder.1.weight/bias)
        self.linear2 = nn.Linear(self.CAP_FEAT_DIM, self.HIDDEN_DIM, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        """Project text embeddings to transformer dimension.

        Args:
            x: Text embeddings [B, seq_len, cap_feat_dim] from Qwen3

        Returns:
            Projected embeddings [B, seq_len, hidden_dim]
        """
        x = self.linear1(x)  # Normalize
        x = self.linear2(x)  # Project to hidden_dim
        return x
