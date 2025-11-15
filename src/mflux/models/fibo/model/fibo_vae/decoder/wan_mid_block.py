import mlx.core as mx
from mlx import nn

from mflux.models.fibo.model.fibo_vae.decoder.wan_attention_block import WanAttentionBlock
from mflux.models.fibo.model.fibo_vae.decoder.wan_residual_block import WanResidualBlock


class WanMidBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0, non_linearity: str = "silu", num_layers: int = 1):
        super().__init__()
        self.resnets = [WanResidualBlock(dim, dim, dropout, non_linearity)]
        self.attentions = []
        for _ in range(num_layers):
            self.attentions.append(WanAttentionBlock(dim))
            self.resnets.append(WanResidualBlock(dim, dim, dropout, non_linearity))

    def __call__(self, x: mx.array) -> mx.array:
        x = self.resnets[0](x)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            x = attn(x)
            x = resnet(x)
        return x
