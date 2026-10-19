import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux3.model.flux3_transformer.flux3_attention import Flux3Attention
from mflux.models.flux3.model.flux3_transformer.flux3_timestep_embedder import Flux3TimestepEmbedder
from mflux.models.flux3.model.flux3_transformer.flux3_transformer import Flux3Transformer

pytestmark = pytest.mark.fast
FIXTURE = Path(__file__).parents[1] / "resources" / "flux3_transformer_parity.npz"
PROFILES = ("fp32", "bf16_timestep", "bf16")
CASES = [
    (kind, profile, case)
    for kind in ("distilled", "base")
    for profile in PROFILES
    for case in ("txt2img", "reference", "multi_reference", "target_zero")
    if kind == "distilled" or case != "target_zero"
]


@pytest.fixture(scope="module")
def reference_data():
    with np.load(FIXTURE, allow_pickle=False) as archive:
        return dict(archive)


class Flux3Parity:
    @staticmethod
    def model(
        data: dict[str, np.ndarray], kind: str, profile: str, monkeypatch: pytest.MonkeyPatch
    ) -> Flux3Transformer:
        monkeypatch.setattr(ModelConfig, "precision", mx.bfloat16 if profile == "bf16" else mx.float32)
        if profile == "fp32":
            monkeypatch.setattr(Flux3TimestepEmbedder, "EMBEDDING_DTYPE", mx.float32)
        # Other profiles exercise the production BF16 timestep default without overriding it.
        model = Flux3Transformer(**json.loads(str(data[f"config/{kind}"])))
        prefix = f"weight/{kind}/"
        # Like WeightLoader._convert_precision, BF16 casts every weight, Q/K norms included.
        dtype = mx.bfloat16 if profile == "bf16" else mx.float32
        weights = []
        for key, value in data.items():
            if key.startswith(prefix):
                name = key.removeprefix(prefix).replace(".attn.to_out.0.", ".attn.to_out.")
                weights.append((name, mx.array(value).astype(dtype)))
        model.load_weights(weights, strict=True)
        return model

    @staticmethod
    def inputs(data: dict[str, np.ndarray], kind: str, case: str) -> dict[str, mx.array]:
        args = {key.removeprefix("input/"): mx.array(value) for key, value in data.items() if key.startswith("input/")}
        if kind == "base":
            args.pop("target_timestep")
        elif case == "target_zero":
            args["target_timestep"] = mx.zeros((1,))
        if case in ("reference", "multi_reference"):
            count = 2 if case == "reference" else 4
            args["cond_hidden_states"] = mx.array(data["condition/image_cond_hidden_states"][:, :count])
            args["cond_ids"] = mx.array(data["condition/image_cond_ids"][:count])
        return args


@pytest.mark.parametrize("kind,profile,case", CASES)
def test_transformer_matches_diffusers(reference_data, monkeypatch, kind, profile, case):
    model = Flux3Parity.model(reference_data, kind, profile, monkeypatch)
    actual = np.array(model(**Flux3Parity.inputs(reference_data, kind, case)).astype(mx.float32))
    expected = reference_data[f"expected/{kind}/{profile}/{case}"]
    assert np.isfinite(actual).all()
    # FP32 detects algebra/layout drift; BF16 permits rounding differences between Metal kernels.
    rtol, atol = (2e-2, 2e-2) if profile == "bf16" else (2e-5, 2e-6)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    relative_l2 = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    assert relative_l2 < (0.02 if profile == "bf16" else 1e-5)


@pytest.mark.parametrize("kind", ["distilled", "base"])
def test_target_timestep_contract(reference_data, monkeypatch, kind):
    model = Flux3Parity.model(reference_data, kind, "bf16_timestep", monkeypatch)
    args = Flux3Parity.inputs(reference_data, kind, "txt2img")
    if kind == "distilled":
        args.pop("target_timestep")
    else:
        args["target_timestep"] = mx.zeros((1,))
    with pytest.raises(ValueError, match="target_timestep"):
        model(**args)


@pytest.mark.parametrize("kind", ["distilled", "base"])
def test_bf16_qk_norms_apply_their_weights(reference_data, monkeypatch, kind):
    # Random weights give near-uniform attention, so the transformer tolerance cannot see ignored Q/K scales.
    model = Flux3Parity.model(reference_data, kind, "bf16", monkeypatch)
    attentions = [(name, module) for name, module in model.named_modules() if isinstance(module, Flux3Attention)]
    assert attentions
    for name, attention in attentions:
        x = mx.random.normal((1, attention.heads, 16, attention.dim_head), key=mx.random.key(0)).astype(mx.bfloat16)
        identity = mx.ones((16, attention.dim_head // 2)), mx.zeros((16, attention.dim_head // 2))
        x64 = np.array(x.astype(mx.float32)).astype(np.float64)
        for norm_name in ("norm_q", "norm_k"):
            weight = mx.array(reference_data[f"weight/{kind}/{name}.{norm_name}.weight"]).astype(mx.bfloat16)
            expected = x64 / np.sqrt((x64**2).mean(-1, keepdims=True) + 1e-6) * np.array(weight.astype(mx.float32))
            actual = Flux3Attention._norm_rope(x, getattr(attention, norm_name), *identity)
            assert actual.dtype == mx.bfloat16
            np.testing.assert_allclose(np.array(actual.astype(mx.float32)), expected, rtol=1e-2, atol=1e-6)
