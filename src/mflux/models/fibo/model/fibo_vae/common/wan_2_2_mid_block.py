import mlx.core as mx
from mlx import nn

from mflux.models.fibo.model.fibo_vae.common.wan_2_2_attention_block import Wan2_2_AttentionBlock
from mflux.models.fibo.model.fibo_vae.common.wan_2_2_residual_block import Wan2_2_ResidualBlock


class Wan2_2_MidBlock(nn.Module):
    def __init__(self, dim: int, non_linearity: str = "silu", num_layers: int = 1):
        super().__init__()
        self.resnets = [Wan2_2_ResidualBlock(dim, dim, non_linearity)]
        self.attentions = []
        for _ in range(num_layers):
            self.attentions.append(Wan2_2_AttentionBlock(dim))
            self.resnets.append(Wan2_2_ResidualBlock(dim, dim, non_linearity))

    def __call__(self, x: mx.array) -> mx.array:
        x = self.resnets[0](x)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            x = attn(x)
            x = resnet(x)
        return x
