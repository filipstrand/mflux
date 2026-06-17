import json

import mlx.core as mx
import mlx.nn as nn
import pytest

from mflux.models.ideogram4.ideogram4_initializer import Ideogram4Initializer
from mflux.models.ideogram4.model.ideogram4_transformer.fp8_linear import Fp8Linear

pytestmark = pytest.mark.fast


class TestFp8LinearBf16Fallthrough:
    def test_non_uint8_weight_uses_plain_matmul(self):
        # mlx-forge bf16 checkpoints load real bf16 weights into Fp8Linear.
        # __call__ must skip mx.from_fp8 and do a plain matmul when the weight
        # dtype is not uint8.
        layer = Fp8Linear(in_features=4, out_features=3, bias=True)
        weight = mx.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0]],
            dtype=mx.bfloat16,
        )
        bias = mx.array([0.5, -0.5, 1.0], dtype=mx.bfloat16)
        layer.update({"weight": weight, "bias": bias})

        x = mx.array([[1.0, 1.0, 1.0, 1.0]], dtype=mx.bfloat16)
        out = layer(x)
        mx.eval(out)

        expected = x @ weight.T + bias
        assert mx.allclose(out, expected).item()


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # input dims are multiples of group_size (64), as in the real model
        self.qkv = Fp8Linear(128, 384, bias=False)
        self.o = Fp8Linear(128, 128, bias=True)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = Fp8Linear(64, 128, bias=True)
        self.embed = nn.Embedding(2, 128)  # must NOT be swapped
        self.layers = [_Block(), _Block()]  # nested-in-list path


class TestReplaceFp8WithQuantized:
    def test_swaps_all_fp8_preserves_dims_and_bias(self):
        model = _Model()
        Ideogram4Initializer._replace_fp8_with_quantized(model, bits=8, group_size=64)

        # top-level projection swapped, dims + bias preserved
        assert isinstance(model.proj, nn.QuantizedLinear)
        assert (model.proj.bits, model.proj.group_size) == (8, 64)
        # scales shape (out, in // group_size) confirms both dims preserved: (128, 64 // 64)
        assert model.proj.scales.shape == (128, 1)
        assert "bias" in model.proj

        # nested-in-list modules swapped via leaf_modules/tree_map_with_path
        for block in model.layers:
            assert isinstance(block.qkv, nn.QuantizedLinear)
            assert isinstance(block.o, nn.QuantizedLinear)
            assert "bias" not in block.qkv  # bias=False preserved
            assert "bias" in block.o

        # embeddings are left alone
        assert isinstance(model.embed, nn.Embedding)
        assert not isinstance(model.embed, nn.QuantizedEmbedding)

    def test_no_fp8_modules_remain(self):
        model = _Model()
        Ideogram4Initializer._replace_fp8_with_quantized(model, bits=8)
        remaining = [m for _, m in model.named_modules() if isinstance(m, Fp8Linear)]
        assert remaining == []


class TestMlxForgeDetection:
    def test_is_mlx_forge_true_when_split_model_present(self, tmp_path):
        (tmp_path / "split_model.json").write_text(json.dumps({"format": "split"}))
        assert Ideogram4Initializer._is_mlx_forge(tmp_path) is True

    def test_is_mlx_forge_false_when_absent(self, tmp_path):
        assert Ideogram4Initializer._is_mlx_forge(tmp_path) is False

    def test_bits_none_when_not_quantized(self, tmp_path):
        (tmp_path / "split_model.json").write_text(json.dumps({"quantized": False}))
        assert Ideogram4Initializer._mlx_forge_bits(tmp_path) is None

    def test_bits_read_when_quantized(self, tmp_path):
        (tmp_path / "split_model.json").write_text(json.dumps({"quantized": True, "quantization_bits": 8}))
        assert Ideogram4Initializer._mlx_forge_bits(tmp_path) == 8
