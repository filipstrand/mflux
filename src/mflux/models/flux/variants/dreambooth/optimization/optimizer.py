from enum import Enum
from pathlib import Path

import mlx.core as mx
import mlx.optimizers
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from mflux.models.flux.variants.dreambooth.state.training_spec import TrainingSpec
from mflux.models.flux.variants.dreambooth.state.zip_util import ZipUtil


class Optimizers(Enum):
    ADAM = ("Adam", optim.Adam)
    ADAMW = ("AdamW", optim.AdamW)

    def __init__(
        self,
        alias: str,
        optimizer: mlx.optimizers.Optimizer,
    ):
        self.alias = alias
        self.optimizer = optimizer

    @staticmethod
    def from_alias(alias: str) -> "mlx.optimizers.Optimizer":
        try:
            for opt in Optimizers:
                if opt.alias == alias:
                    return opt.optimizer
        except KeyError:
            raise ValueError(f"'{alias}' is not a valid optimization")


class Optimizer:
    def __init__(self, optimizer: mlx.optimizers.Optimizer):
        self.optimizer = optimizer

    def save(self, path: Path) -> None:
        state = tree_flatten(self.optimizer.state)
        mx.save_safetensors(str(path), dict(state))

    @staticmethod
    def from_spec(training_spec: TrainingSpec) -> "Optimizer":
        opt = Optimizers.from_alias(training_spec.optimizer.name)
        # noinspection PyCallingNonCallable
        opt = opt(learning_rate=training_spec.optimizer.learning_rate)

        # Load from state if present in the spec
        if training_spec.optimizer.state_path is not None:
            state = ZipUtil.unzip(
                zip_path=training_spec.checkpoint_path,
                filename=training_spec.optimizer.state_path,
                loader=lambda x: tree_unflatten(list(mx.load(x).items())),
            )
            opt.state = state

        return Optimizer(opt)
