import mlx.core as mx
import numpy as np
import PIL.Image
import pytest
from mlx import nn

from mflux.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.krea2.krea2_initializer import Krea2Initializer
from mflux.models.krea2.model.krea2_transformer.transformer import Krea2Transformer
from mflux.models.krea2.variants.controlnet import krea2_controlnet
from mflux.models.krea2.variants.controlnet.krea2_controlnet_util import Krea2ControlnetUtil
from mflux.models.krea2.weights.krea2_weight_definition import Krea2WeightDefinition

# A small Krea 2 transformer with the real submodule names, so these tests need no download.
TINY = dict(features=64, heads=4, kvheads=4, layers=2, txtlayers=2, txtdim=16, txtheads=2, txtkvheads=2, patch=2, channels=16)  # fmt: off
IMAGE_IN = TINY["channels"] * TINY["patch"] ** 2
RANK = 4


def _inputs():
    hidden = mx.random.normal((1, 16, 8, 8))
    control = mx.random.normal((1, 16, 8, 8))
    context = mx.random.normal((1, 3, TINY["txtlayers"] * TINY["txtdim"]))
    timestep = mx.array([0.5])
    return hidden, control, context, timestep


def _checkpoint(transformer: Krea2Transformer) -> dict[str, mx.array]:
    # Every key the depth ControlNet holds, at the tiny transformer's shapes.
    control = {
        "first.weight": mx.random.normal((TINY["features"], 2 * IMAGE_IN)),
        "first.bias": mx.random.normal((TINY["features"],)),
    }
    for i, block in enumerate(transformer.blocks):
        for target in Krea2Initializer._CONTROL_TARGETS:
            sub, attr = target.split(".")
            out_features, in_features = getattr(getattr(block, sub), attr).weight.shape
            control[f"blocks.{i}.{target}.A"] = mx.random.normal((RANK, in_features))
            control[f"blocks.{i}.{target}.B"] = mx.random.normal((out_features, RANK))
    return control


def _apply(tmp_path, transformer: Krea2Transformer, control: dict[str, mx.array], strength: float = 1.0) -> None:
    path = tmp_path / "control.safetensors"
    mx.save_safetensors(str(path), control)
    model = type("M", (), {})()
    model.transformer = transformer
    Krea2Initializer._apply_control_checkpoint(model, str(path), strength=strength)


def test_control_changes_the_prediction():
    t = Krea2Transformer(**TINY)
    t.control_first = nn.Linear(IMAGE_IN, TINY["features"], bias=False)
    hidden, control, context, timestep = _inputs()
    out = t(hidden, timestep, context, control=control)
    assert out.shape == hidden.shape
    other = t(hidden, timestep, context, control=mx.random.normal((1, 16, 8, 8)))
    assert not mx.allclose(out, other)


def test_base_path_runs_without_control():
    t = Krea2Transformer(**TINY)
    hidden, _, context, timestep = _inputs()
    assert t(hidden, timestep, context).shape == hidden.shape


@pytest.mark.parametrize("strength", [1.0, 0.5])
def test_merge_splits_first_and_adds_every_delta(tmp_path, strength):
    t = Krea2Transformer(**TINY)
    before = {(i, target): mx.array(getattr(getattr(block, target.split(".")[0]), target.split(".")[1]).weight)
              for i, block in enumerate(t.blocks) for target in Krea2Initializer._CONTROL_TARGETS}  # fmt: off
    control = _checkpoint(t)
    _apply(tmp_path, t, control, strength)

    # first keeps its shape and takes the image columns; control_first takes the depth columns.
    assert tuple(t.first.weight.shape) == (TINY["features"], IMAGE_IN)
    assert mx.allclose(t.first.weight, control["first.weight"][:, :IMAGE_IN])
    assert mx.allclose(t.first.bias, control["first.bias"])
    assert mx.allclose(t.control_first.weight, control["first.weight"][:, IMAGE_IN:])
    for (i, target), weight in before.items():
        sub, attr = target.split(".")
        delta = control[f"blocks.{i}.{target}.B"] @ control[f"blocks.{i}.{target}.A"]
        merged = getattr(getattr(t.blocks[i], sub), attr).weight
        assert mx.allclose(merged, weight + strength * delta, atol=1e-4), (i, target)


def test_split_projection_matches_the_reference_concat():
    # first(img) + control_first(ctrl) must equal the reference's widened Linear on cat([img, ctrl]).
    wide = nn.Linear(2 * IMAGE_IN, TINY["features"], bias=True)
    img, ctrl = mx.random.normal((1, 16, IMAGE_IN)), mx.random.normal((1, 16, IMAGE_IN))
    first = nn.Linear(IMAGE_IN, TINY["features"], bias=True)
    first.weight, first.bias = wide.weight[:, :IMAGE_IN], wide.bias
    control_first = nn.Linear(IMAGE_IN, TINY["features"], bias=False)
    control_first.weight = wide.weight[:, IMAGE_IN:]
    assert mx.allclose(first(img) + control_first(ctrl), wide(mx.concatenate([img, ctrl], axis=-1)), atol=1e-5)


@pytest.mark.parametrize(
    "corrupt, match",
    [
        (lambda c: c.update({"first.weight": mx.zeros((TINY["features"], 999))}), "Control 'first'"),
        (lambda c: c.pop("blocks.1.mlp.down.B"), "1 missing keys"),
        (lambda c: c.update({"blocks.0.attn.extra.A": mx.zeros((RANK, 4))}), "1 unknown keys"),
    ],
    ids=["wrong-first-shape", "missing-pair", "unknown-key"],
)
def test_a_wrong_checkpoint_fails_before_anything_changes(tmp_path, corrupt, match):
    t = Krea2Transformer(**TINY)
    first_weight = mx.array(t.first.weight)
    wq = mx.array(t.blocks[0].attn.wq.weight)
    control = _checkpoint(t)
    corrupt(control)
    with pytest.raises(ValueError, match=match):
        _apply(tmp_path, t, control)
    assert mx.array_equal(t.first.weight, first_weight)
    assert mx.array_equal(t.blocks[0].attn.wq.weight, wq)
    assert "control_first" not in t


def test_depth_pro_runs_once_per_source_image(tmp_path, monkeypatch):
    # Several seeds on one photo reuse the estimate; a different photo is estimated again.
    calls = []

    class FakeDepthPro:
        def create_depth_map(self, image_path):
            calls.append(image_path)
            return type("R", (), {"depth_image": PIL.Image.new("L", (8, 8))})()

    monkeypatch.setattr(krea2_controlnet, "DepthPro", FakeDepthPro)
    model = krea2_controlnet.Krea2Controlnet.__new__(krea2_controlnet.Krea2Controlnet)
    model.depth_pro = None
    model._estimated_depth = None
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    model._depth_image(a, None)
    model._depth_image(a, None)
    model._depth_image(b, None)
    assert calls == [a, b]


def test_a_16_bit_depth_map_keeps_its_range():
    # A plain convert("RGB") clips 16-bit values to white; the map must keep its near-to-far gradient.
    gradient = np.tile(np.linspace(0, 65535, 16, dtype=np.uint16), (4, 1))
    rgb = np.asarray(Krea2ControlnetUtil._to_rgb(PIL.Image.fromarray(gradient)))
    assert rgb.shape == (4, 16, 3)
    assert rgb[0, 0, 0] == 0 and rgb[0, -1, 0] == 255
    assert 100 < rgb[0, 8, 0] < 160


def test_a_16_bit_depth_map_file_keeps_its_range(tmp_path):
    # Through the file path, not only in memory: the loader must not clip the map to RGB first.
    path = tmp_path / "depth16.png"
    PIL.Image.fromarray(np.tile(np.linspace(0, 65535, 16, dtype=np.uint16), (4, 1))).save(path)
    image = Krea2ControlnetUtil.get_depth_image(None, None, path)
    rgb = np.asarray(Krea2ControlnetUtil._to_rgb(image))
    assert rgb[0, 0, 0] == 0 and rgb[0, -1, 0] == 255
    assert 100 < rgb[0, 8, 0] < 160


def test_a_16_bit_map_matches_the_8_bit_map_it_was_made_from():
    # Values are scaled, not stretched: a map that never reaches black or white stays that way.
    gray = np.array([[30, 100, 220]], dtype=np.uint8)
    as16 = PIL.Image.fromarray(gray.astype(np.uint16) * 257)
    assert np.array_equal(np.asarray(Krea2ControlnetUtil._to_rgb(as16))[..., 0], gray)


def _live_control(tmp_path, transformer: Krea2Transformer, control: dict[str, mx.array], strength: float = 1.0) -> None:
    # The quantized-save route: first is replaced, the deltas stay adapters.
    path = tmp_path / "control.safetensors"
    mx.save_safetensors(str(path), control)
    first_w, first_b, pairs = Krea2Initializer._load_control_checkpoint(transformer, str(path))
    Krea2Initializer._set_control_first(transformer, first_w, first_b)
    Krea2Initializer._attach_control_adapters(transformer, pairs, strength)


def test_live_adapters_match_the_merge(tmp_path):
    # On a quantized save the deltas run as adapters; on the same base they must give what the merge gives.
    merged = Krea2Transformer(**TINY)
    live = Krea2Transformer(**TINY)
    live.update(merged.parameters())
    control = _checkpoint(merged)
    _apply(tmp_path, merged, control, strength=0.5)
    _live_control(tmp_path, live, control, strength=0.5)
    hidden, depth, context, timestep = _inputs()
    expected = merged(hidden, timestep, context, control=depth)
    assert mx.allclose(live(hidden, timestep, context, control=depth), expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("bits", [8, 6, 4])
def test_a_quantized_save_takes_the_control_as_adapters(tmp_path, bits):
    # 6 bits does not divide 32: the packed width must still give the real input size.
    t = Krea2Transformer(**TINY)
    control = _checkpoint(t)
    nn.quantize(t, bits=bits, class_predicate=Krea2WeightDefinition.quantization_predicate)
    _live_control(tmp_path, t, control)
    assert type(t.first) is nn.Linear
    assert mx.allclose(t.first.weight, control["first.weight"][:, :IMAGE_IN].astype(t.first.weight.dtype))
    wq = t.blocks[0].attn.wq
    assert isinstance(wq, LoRALinear) and isinstance(wq.linear, nn.QuantizedLinear)
    hidden, depth, context, timestep = _inputs()
    assert t(hidden, timestep, context, control=depth).shape == hidden.shape


def test_control_adapters_stack_on_a_live_lora(tmp_path):
    # The init order on a quantized save: load and check the file, replace first, apply the LoRAs, add the deltas.
    t = Krea2Transformer(**TINY)
    path = tmp_path / "control.safetensors"
    mx.save_safetensors(str(path), _checkpoint(t))
    first_w, first_b, pairs = Krea2Initializer._load_control_checkpoint(t, str(path))
    Krea2Initializer._set_control_first(t, first_w, first_b)
    out_features, in_features = t.blocks[0].attn.wq.weight.shape
    lora = {"lora_A": mx.random.normal((in_features, RANK)), "lora_B": mx.random.normal((RANK, out_features))}
    LoRALoader._apply_lora_matrices_to_target(t, "blocks.0.attn.wq", lora, scale=1.0, role=None)
    Krea2Initializer._attach_control_adapters(t, pairs, strength=1.0)
    wq = t.blocks[0].attn.wq
    assert isinstance(wq, FusedLoRALinear) and len(wq.loras) == 2
