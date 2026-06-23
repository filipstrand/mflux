import mlx.core as mx
from mlx import nn

from mflux.models.common_models.qwen3_vl.qwen3_vl_decoder import Qwen3VLDecoder


class Krea2TextEncoder(nn.Module):
    def __init__(
        self,
        select_layers: tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35),
    ) -> None:
        super().__init__()
        self.encoder = Qwen3VLDecoder()
        self.select_layers = select_layers

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
        position_ids: mx.array,
        trim_start: int,
    ) -> tuple[mx.array, mx.array]:
        _, hidden_states = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            return_logits=False,
        )
        selected = mx.stack([hidden_states[i] for i in self.select_layers], axis=2)
        return selected[:, trim_start:], attention_mask[:, trim_start:]
