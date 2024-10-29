from enum import Enum
from pathlib import Path

import mlx.core as mx
import mlx.optimizers
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from mflux.dreambooth.state.training_spec import OptimizerSpec


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
    def from_spec(optimizer_spec: OptimizerSpec) -> "Optimizer":
        opt = Optimizers.from_alias(optimizer_spec.name)
        # noinspection PyCallingNonCallable
        opt = opt(learning_rate=optimizer_spec.learning_rate)

        # Load from state if present in the spec
        if optimizer_spec.state_path is not None:
            state = tree_unflatten(list(mx.load(optimizer_spec.state_path).items()))
            opt.state = state

        return Optimizer(opt)
