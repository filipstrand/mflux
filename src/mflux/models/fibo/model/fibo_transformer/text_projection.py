from mlx import nn


class BriaFiboTextProjection(nn.Module):
    def __init__(self, in_features: int = 2048, hidden_size: int = 1536):
        super().__init__()
        self.linear = nn.Linear(in_features, hidden_size, bias=False)

    def __call__(self, caption):
        return self.linear(caption)
