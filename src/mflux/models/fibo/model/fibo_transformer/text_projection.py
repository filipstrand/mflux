from mlx import nn


class BriaFiboTextProjection(nn.Module):
    """
    MLX port of diffusers.models.transformers.transformer_bria_fibo.BriaFiboTextProjection.

    Projects text encoder layer outputs to half of the transformer inner_dim.
    """

    def __init__(self, in_features: int, hidden_size: int):
        super().__init__()
        # MLX Linear uses positional arguments: nn.Linear(in_dim, out_dim, bias=True)
        self.linear = nn.Linear(in_features, hidden_size, bias=False)

    def __call__(self, caption):
        return self.linear(caption)
