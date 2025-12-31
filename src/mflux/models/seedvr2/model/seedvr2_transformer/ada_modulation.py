import mlx.core as mx
from mlx import nn


class AdaModulation(nn.Module):
    def __init__(self, dim: int, shared_weights: bool = False, is_last_layer: bool = False):
        super().__init__()
        self.shared_weights = shared_weights
        self.is_last_layer = is_last_layer

        if shared_weights:
            self.params_all = AdaModulation._init_params(dim)
        else:
            self.params_vid = AdaModulation._init_params(dim)
            if not is_last_layer:
                self.params_txt = AdaModulation._init_params(dim)

    @staticmethod
    def _init_params(dim: int) -> dict[str, mx.array]:
        return {
            "attn_shift": mx.zeros((dim,)),
            "attn_scale": mx.ones((dim,)),
            "attn_gate": mx.zeros((dim,)),
            "mlp_shift": mx.zeros((dim,)),
            "mlp_scale": mx.ones((dim,)),
            "mlp_gate": mx.zeros((dim,)),
        }

    def modulate_vid(
        self,
        hidden: mx.array,
        emb: mx.array,
        layer: str,
        mode: str,
    ) -> mx.array:
        params = self.params_all if self.shared_weights else self.params_vid
        return AdaModulation._apply_modulation(hidden, emb, params, layer, mode)

    def modulate_txt(
        self,
        hidden: mx.array,
        emb: mx.array,
        layer: str,
        mode: str,
    ) -> mx.array:
        if self.is_last_layer:
            return hidden
        params = self.params_all if self.shared_weights else self.params_txt
        return AdaModulation._apply_modulation(hidden, emb, params, layer, mode)

    @staticmethod
    def _apply_modulation(
        hidden: mx.array,
        emb: mx.array,
        params: dict[str, mx.array],
        layer: str,
        mode: str,
    ) -> mx.array:
        layer_idx = 0 if layer == "attn" else 1
        mod = emb[:, :, layer_idx, :]

        if mode == "in":
            shift = mod[..., 0][:, None] + params[f"{layer}_shift"]
            scale = mod[..., 1][:, None] + params[f"{layer}_scale"]
            return hidden * scale + shift
        else:
            gate = mod[..., 2][:, None] + params[f"{layer}_gate"]
            return hidden * gate
