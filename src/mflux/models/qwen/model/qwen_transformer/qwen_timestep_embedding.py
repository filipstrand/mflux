import mlx.core as mx
from mlx import nn


class QwenTimestepEmbedding(nn.Module):
    """
    Matches PyTorch TimestepEmbedding class exactly (diffusers/models/embeddings.py:1262-1307).

    PyTorch initialization:
        TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        # Uses default act_fn="silu", no cond_proj, no post_act
    """

    def __init__(self, proj_dim: int, inner_dim: int):
        super().__init__()
        # PyTorch: self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)
        # sample_proj_bias defaults to True
        self.linear_1 = nn.Linear(proj_dim, inner_dim, bias=True)

        # PyTorch: self.act = get_activation(act_fn) where act_fn="silu" (default)
        # PyTorch: self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)
        self.linear_2 = nn.Linear(inner_dim, inner_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Matches PyTorch TimestepEmbedding.forward exactly.

        PyTorch:
            sample = self.linear_1(sample)
            if self.act is not None: sample = self.act(sample)
            sample = self.linear_2(sample)
            return sample
        """
        # PyTorch: sample = self.linear_1(sample)
        x = self.linear_1(x)

        # PyTorch: if self.act is not None: sample = self.act(sample)
        # act_fn="silu" (default) -> nn.silu
        x = nn.silu(x)

        # PyTorch: sample = self.linear_2(sample)
        x = self.linear_2(x)

        return x
