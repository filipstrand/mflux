from mlx import nn

from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.layer.lokr_linear_layer import LoKrLinear


class FusedLoRALinear(nn.Module):
    """A shared base Linear plus a list of stacked adapters applied as summed
    residuals. ``loras`` may hold a MIX of :class:`LoRALinear` and :class:`LoKrLinear`
    — each contributes ``adapter.residual(x)``, so multiple LoRA and/or LoKr adapters
    compose on a single module. The base is applied exactly once.
    """

    def __init__(self, base_linear: nn.Linear | nn.QuantizedLinear, loras: list):
        super().__init__()
        self.base_linear = base_linear
        self.loras = loras  # LoRALinear and/or LoKrLinear, applied as summed residuals

    def __call__(self, x):
        out = self.base_linear(x)
        for adapter in self.loras:
            out = out + adapter.residual(x)
        return out

    @staticmethod
    def unwrap(module) -> tuple:
        """Return ``(base_linear, [adapters])`` for whatever is installed at a target:
        a plain Linear (``[]``), a single LoRALinear/LoKrLinear, or a FusedLoRALinear.
        Lets either loader stack a new adapter onto any existing one instead of
        replacing or skipping it."""
        if isinstance(module, FusedLoRALinear):
            return module.base_linear, list(module.loras)
        if isinstance(module, (LoRALinear, LoKrLinear)):
            return module.linear, [module]
        return module, []
