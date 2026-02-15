import mlx.core as mx
from mlx import nn

from mflux.models.seedvr2.model.seedvr2_transformer.swiglu_mlp import GELUMLP, SwiGLUMLP


class MMSwiGLU(nn.Module):
    def __init__(
        self,
        vid_dim: int,
        txt_dim: int,
        expand_ratio: int = 4,
        shared_weights: bool = False,
        is_last_layer: bool = False,
        mlp_type: str = "swiglu",
    ):
        super().__init__()
        self.shared_weights = shared_weights
        self.is_last_layer = is_last_layer
        self.mlp_type = mlp_type

        mlp_cls = SwiGLUMLP
        mlp_kwargs = {"expand_ratio": expand_ratio}
        if mlp_type == "normal":
            mlp_cls = GELUMLP
            mlp_kwargs["bias"] = True

        if shared_weights:
            self.all = mlp_cls(dim=vid_dim, **mlp_kwargs)
        else:
            self.vid = mlp_cls(dim=vid_dim, **mlp_kwargs)
            if not is_last_layer:
                self.txt = mlp_cls(dim=txt_dim, **mlp_kwargs)

    def __call__(self, vid: mx.array, txt: mx.array) -> tuple[mx.array, mx.array]:
        if self.shared_weights:
            vid_out = self.all(vid)
            txt_out = self.all(txt)
        else:
            vid_out = self.vid(vid)
            txt_out = self.txt(txt) if not self.is_last_layer else txt

        return vid_out, txt_out
