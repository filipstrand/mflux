import mlx.core as mx
import pytest
from mlx import nn

from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.lora.mapping.lora_saver import LoRASaver

pytestmark = pytest.mark.fast

OUT, IN = 64, 128


class _Block(nn.Module):
    def __init__(self, dtype: mx.Dtype, size: float):
        super().__init__()
        mx.random.seed(0)
        self.proj = nn.Linear(IN, OUT, bias=False)
        self.proj.weight = mx.random.normal((OUT, IN)).astype(dtype)
        lora = LoRALinear.from_linear(self.proj, r=8)
        # size is the delta against the weights: 1e-3 sits under half a bfloat16 step (2**-9).
        lora.lora_A = mx.random.normal((IN, 8)) * (size / 8) ** 0.5
        lora.lora_B = mx.random.normal((8, OUT)) * (size / 8) ** 0.5
        self.proj = lora


@pytest.mark.parametrize(
    ("dtype", "size", "low", "high"),
    [(mx.float32, 1e-3, 0.999, 1.001), (mx.bfloat16, 0.1, 0.95, 1.05), (mx.bfloat16, 1e-3, 0.0, 0.5)],
)
def test_bake_reports_the_share_of_the_update_the_weights_hold(dtype, size, low, high):
    kept = []
    LoRASaver.bake_and_strip_lora(_Block(dtype, size), kept=kept)

    assert len(kept) == 1
    assert low < kept[0] < high


@pytest.mark.parametrize(
    ("bits", "stored", "size", "low", "high"),
    [
        # Folded onto the decoded q8 grid, then the same from the dense weights -q quantized.
        (8, False, 0.1, 0.9, 1.1),
        (8, False, 1e-3, 0.0, 0.5),
        (8, True, 0.1, 0.9, 1.1),
        # A q4 layer is re-quantized at q8, so a large update survives it.
        (4, False, 0.1, 0.9, 1.1),
    ],
)
def test_bake_reports_the_share_on_quantized_layers(bits, stored, size, low, high):
    block = _Block(mx.bfloat16, size)
    weight = block.proj.linear.weight
    block.proj.linear = block.proj.linear.to_quantized(group_size=64, bits=bits)
    kept = []

    dense = {"proj": {"weight": weight}} if stored else None
    LoRASaver.bake_and_strip_lora(block, dense_weights=dense, kept=kept)

    assert isinstance(block.proj, nn.QuantizedLinear)
    assert len(kept) == 1
    assert low < kept[0] < high


@pytest.mark.parametrize(("size", "warned"), [(1e-3, True), (0.1, False)])
def test_loader_says_how_to_keep_an_update_the_bake_rounds_away(monkeypatch, capsys, size, warned):
    # The viggle_turbo LoRA for Qwen-Image-2.1 kept 66% of itself baked into bfloat16 (#835).
    # _Block arrives with its adapter already in place, so applying the file does nothing here.
    monkeypatch.setattr(LoRALoader, "_apply_single_lora", staticmethod(lambda *args, **kwargs: None))
    monkeypatch.setattr(
        "mflux.models.common.lora.mapping.lora_loader.LoraResolution.resolve_paths", lambda paths: paths
    )

    LoRALoader.load_and_apply_lora(lora_mapping=[], transformer=_Block(mx.bfloat16, size), lora_paths=["adapter"])

    assert ("--no-bake-lora" in capsys.readouterr().out) == warned
