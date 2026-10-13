import mlx.core as mx
import pytest
from mlx import nn

from mflux.models.krea2.krea2_initializer import Krea2Initializer
from mflux.models.krea2.model.krea2_transformer.transformer import Krea2Transformer

# A small Krea 2 transformer with the real submodule names, so these tests need no download.
TINY = dict(features=64, heads=4, kvheads=4, layers=2, txtlayers=2, txtdim=16, txtheads=2, txtkvheads=2, patch=2, channels=16)  # fmt: off


def _inputs():
    hidden = mx.random.normal((1, 16, 8, 8))
    control = mx.random.normal((1, 16, 8, 8))
    context = mx.random.normal((1, 3, TINY["txtlayers"] * TINY["txtdim"]))
    timestep = mx.array([0.5])
    return hidden, control, context, timestep


def test_control_concat_shape_and_effect():
    t = Krea2Transformer(**TINY)
    # The depth half of the reference's widened input projection.
    t.control_first = nn.Linear(64, TINY["features"], bias=False)
    hidden, control, context, timestep = _inputs()

    out = t(hidden, timestep, context, control=control)
    assert out.shape == hidden.shape

    # A different control latent must change the prediction (the depth signal is actually used).
    other = t(hidden, timestep, context, control=mx.random.normal((1, 16, 8, 8)))
    assert not mx.allclose(out, other)


def test_base_path_unaffected_by_control_param():
    # Without control, the transformer keeps its narrow first projection and behaves as plain txt2img.
    t = Krea2Transformer(**TINY)
    hidden, _, context, timestep = _inputs()
    out = t(hidden, timestep, context)
    assert out.shape == hidden.shape


def test_apply_control_checkpoint_merges_first_and_deltas(tmp_path):
    t = Krea2Transformer(**TINY)
    features, patch, channels = TINY["features"], TINY["patch"], TINY["channels"]
    narrow_in = channels * patch * patch  # 64
    wide_in = 2 * narrow_in  # 128
    rank = 4

    # Snapshot a base attention weight to verify the delta is added on top of it.
    base_wq = mx.array(t.blocks[0].attn.wq.weight)
    out_wq = t.blocks[0].attn.wq.weight.shape[0]

    control = {
        "first.weight": mx.random.normal((features, wide_in)),
        "first.bias": mx.random.normal((features,)),
        "blocks.0.attn.wq.A": mx.random.normal((rank, features)),
        "blocks.0.attn.wq.B": mx.random.normal((out_wq, rank)),
    }
    path = tmp_path / "control.safetensors"
    mx.save_safetensors(str(path), control)

    model = type("M", (), {})()
    model.transformer = t
    Krea2Initializer._apply_control_checkpoint(model, str(path), strength=1.0)

    # first keeps its shape and takes the image columns; control_first takes the depth columns.
    assert tuple(t.first.weight.shape) == (features, narrow_in)
    assert mx.allclose(t.first.weight, control["first.weight"][:, :narrow_in])
    assert mx.allclose(t.first.bias, control["first.bias"])
    assert mx.allclose(t.control_first.weight, control["first.weight"][:, narrow_in:])

    # attn.wq weight gained exactly B @ A at strength 1.0.
    expected = base_wq + control["blocks.0.attn.wq.B"] @ control["blocks.0.attn.wq.A"]
    assert mx.allclose(t.blocks[0].attn.wq.weight, expected, atol=1e-5)


def test_apply_control_checkpoint_rejects_malformed_first(tmp_path):
    t = Krea2Transformer(**TINY)
    control = {
        "first.weight": mx.random.normal((TINY["features"], 999)),  # wrong widened input size
        "first.bias": mx.zeros((TINY["features"],)),
    }
    path = tmp_path / "bad.safetensors"
    mx.save_safetensors(str(path), control)
    model = type("M", (), {})()
    model.transformer = t
    with pytest.raises(ValueError, match="Control 'first'"):
        Krea2Initializer._apply_control_checkpoint(model, str(path), strength=1.0)


def test_apply_control_checkpoint_strength_scales_delta(tmp_path):
    t = Krea2Transformer(**TINY)
    features = TINY["features"]
    rank = 4
    base_wq = mx.array(t.blocks[0].attn.wq.weight)
    out_wq = t.blocks[0].attn.wq.weight.shape[0]
    control = {
        "first.weight": mx.random.normal((features, 128)),
        "first.bias": mx.zeros((features,)),
        "blocks.0.attn.wq.A": mx.random.normal((rank, features)),
        "blocks.0.attn.wq.B": mx.random.normal((out_wq, rank)),
    }
    path = tmp_path / "control.safetensors"
    mx.save_safetensors(str(path), control)

    model = type("M", (), {})()
    model.transformer = t
    Krea2Initializer._apply_control_checkpoint(model, str(path), strength=0.5)

    expected = base_wq + 0.5 * (control["blocks.0.attn.wq.B"] @ control["blocks.0.attn.wq.A"])
    assert mx.allclose(t.blocks[0].attn.wq.weight, expected, atol=1e-5)


def test_apply_control_checkpoint_rejects_a_file_without_deltas(tmp_path):
    # A file with only the input projection is not the depth ControlNet; nothing in the model changes.
    t = Krea2Transformer(**TINY)
    first = t.first
    control = {"first.weight": mx.random.normal((TINY["features"], 128)), "first.bias": mx.zeros((TINY["features"],))}
    path = tmp_path / "first_only.safetensors"
    mx.save_safetensors(str(path), control)
    model = type("M", (), {})()
    model.transformer = t
    with pytest.raises(ValueError, match="no attention/MLP deltas"):
        Krea2Initializer._apply_control_checkpoint(model, str(path), strength=1.0)
    assert t.first is first


def test_split_projection_matches_the_reference_concat():
    # first(img) + control_first(ctrl) must equal the reference's widened Linear on cat([img, ctrl]).
    features, image_in = TINY["features"], TINY["channels"] * TINY["patch"] ** 2
    wide = nn.Linear(2 * image_in, features, bias=True)
    img, ctrl = mx.random.normal((1, 16, image_in)), mx.random.normal((1, 16, image_in))
    first = nn.Linear(image_in, features, bias=True)
    first.weight, first.bias = wide.weight[:, :image_in], wide.bias
    control_first = nn.Linear(image_in, features, bias=False)
    control_first.weight = wide.weight[:, image_in:]
    assert mx.allclose(first(img) + control_first(ctrl), wide(mx.concatenate([img, ctrl], axis=-1)), atol=1e-5)
