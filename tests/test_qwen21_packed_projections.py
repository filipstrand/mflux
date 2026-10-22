from contextlib import suppress

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from mlx.utils import tree_flatten

from mflux.models.common.compute_precision import ComputePrecision
from mflux.models.common.config import ModelConfig
from mflux.models.common.config.config import Config
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.qwen21.model.qwen21_transformer.qwen21_layout import QwenImage21Layout
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer

pytestmark = pytest.mark.fast


class TestQwen21PackedProjections:
    CONFIG = dict(
        in_channels=4,
        out_channels=4,
        context_in_dim=16,
        num_layers=2,
        num_attention_heads=1,
        attention_head_dim=64,
        mlp_ratio=1,
        axes_dims_rope=(16, 24, 24),
    )

    @pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
    @pytest.mark.parametrize("text_cache", [False, True])
    def test_packed_denoise_matches_original_projections(self, dtype, text_cache):
        original = self._model(dtype, packed=False)
        packed = self._model(dtype)
        original.use_text_cache = packed.use_text_cache = text_cache
        inputs = self._inputs(dtype)
        expected = self._generate(original, inputs)
        with packed.inference_projections():
            for block in packed.transformer_blocks:
                assert block.attn._qkv_weight is not None
                assert block.img_mlp._gate_proj_weight is not None
            actual = [self._step(packed, inputs, timestep) for timestep in (1, 2)]
        for block in packed.transformer_blocks:
            assert block.attn._qkv_weight is None
            assert block.img_mlp._gate_proj_weight is None
        tolerance = 2e-2 if dtype == mx.bfloat16 else 2e-5
        for result, reference in zip(actual, expected):
            self._assert_close(result, reference, tolerance)

    @pytest.mark.parametrize("interrupted", [False, True])
    def test_next_generation_observes_in_place_parameter_updates(self, interrupted):
        original, packed = self._model(packed=False), self._model()
        inputs = self._inputs()
        before = self._generate(original, inputs)[-1]
        with suppress(InterruptedError), packed.inference_projections():
            actual = self._step(packed, inputs, 2)
            self._assert_close(actual, before)
            if interrupted:
                raise InterruptedError
        for model in (original, packed):
            block = model.transformer_blocks[0]
            # Same Python array object, changed MLX descriptor: id-based caches miss this.
            block.attn.to_v.weight[:] = -0.5 * block.attn.to_v.weight
            block.img_mlp.gate_layer.weight[:] = 0.5 * block.img_mlp.gate_layer.weight + 0.1
        expected = self._generate(original, inputs)[-1]
        actual = self._generate(packed, inputs)[-1]
        self._assert_close(actual, expected)
        assert not np.allclose(self._numpy(actual), self._numpy(before), atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("fallback", ["disabled", "adapters"])
    @pytest.mark.parametrize("text_cache", [False, True])
    @pytest.mark.parametrize("interrupted", [False, True])
    def test_unpacked_generation_retraces_after_parameter_updates(self, fallback, text_cache, interrupted):
        original, reused = self._model(packed=False), self._model(packed=fallback != "disabled")
        for model in (original, reused):
            model.use_text_cache = text_cache
            if fallback == "adapters":
                for block in model.transformer_blocks:
                    block.attn.to_v = self._adapter(block.attn.to_v)
                    block.img_mlp.gate_layer = self._adapter(block.img_mlp.gate_layer)
        inputs = self._inputs()
        with suppress(InterruptedError), reused.inference_projections():
            for block in reused.transformer_blocks:
                assert block.attn._qkv_weight is None
                assert block.img_mlp._gate_proj_weight is None
            before = self._step(reused, inputs, 2)
            if interrupted:
                raise InterruptedError
        for model in (original, reused):
            model.proj_out.weight[:] = -0.5 * model.proj_out.weight
            if fallback == "adapters":
                for block in model.transformer_blocks:
                    block.attn.to_v.scale = 2.0
                    block.img_mlp.gate_layer.scale = 2.0
        actual = self._generate(reused, inputs)[-1]
        # The reference has never been traced with the previous parameter values.
        expected = self._generate(original, inputs)[-1]
        self._assert_close(actual, expected)
        assert not np.allclose(self._numpy(actual), self._numpy(before), atol=1e-5, rtol=1e-5)

    def test_reloading_weights_after_generation_replaces_packed_values(self):
        original, packed = self._model(packed=False), self._model()
        inputs = self._inputs()
        before = self._generate(packed, inputs)[-1]
        replacement = [(name, value * 0.5) for name, value in tree_flatten(original.parameters())]
        original.load_weights(replacement)
        packed.load_weights(replacement)
        actual = self._generate(packed, inputs)[-1]
        self._assert_close(actual, self._generate(original, inputs)[-1])
        assert not np.allclose(self._numpy(actual), self._numpy(before), atol=1e-5, rtol=1e-5)

    def test_save_inside_inference_scope_reloads_original_checkpoint_schema(self, tmp_path):
        original, packed = self._model(packed=False), self._model()
        inputs = self._inputs()
        expected_parameters = dict(tree_flatten(original.parameters()))
        checkpoint = str(tmp_path / "transformer.safetensors")
        with packed.inference_projections():
            self._step(packed, inputs, 2)
            packed.save_weights(checkpoint)
        saved = mx.load(checkpoint)
        assert saved.keys() == expected_parameters.keys()
        for name, weight in saved.items():
            np.testing.assert_array_equal(self._numpy(weight), self._numpy(expected_parameters[name]))
        restored = self._model()
        restored.load_weights(checkpoint, strict=True)
        self._assert_close(self._generate(restored, inputs)[-1], self._generate(original, inputs)[-1])

    def test_quantized_projection_groups_keep_quantized_math(self):
        original, packed = self._model(packed=False), self._model()
        for model in (original, packed):
            block = model.transformer_blocks[0]
            block.attn.to_q = block.attn.to_q.to_quantized(group_size=64, bits=4)
            block.img_mlp.proj = block.img_mlp.proj.to_quantized(group_size=64, bits=4)
        inputs = self._inputs()
        self._assert_close(self._generate(packed, inputs)[-1], self._generate(original, inputs)[-1])

    def test_live_adapters_and_scale_changes_survive_generation_boundaries(self):
        original, packed = self._model(packed=False), self._model()
        inputs = self._inputs()
        self._generate(packed, inputs)
        for model in (original, packed):
            block = model.transformer_blocks[0]
            block.attn.to_v = self._adapter(block.attn.to_v)
            block.img_mlp.gate_layer = self._adapter(block.img_mlp.gate_layer)
        before = None
        for scale in (0.5, 2.0):
            for model in (original, packed):
                block = model.transformer_blocks[0]
                block.attn.to_v.scale = scale
                block.img_mlp.gate_layer.scale = scale
            actual = self._generate(packed, inputs)[-1]
            self._assert_close(actual, self._generate(original, inputs)[-1])
            if before is not None:
                assert not np.allclose(self._numpy(actual), self._numpy(before), atol=1e-5, rtol=1e-5)
            before = actual

    @pytest.mark.parametrize("conversion", ["storage", "compute"])
    def test_precision_conversion_after_packing_uses_converted_weights(self, conversion):
        original, packed = self._model(mx.bfloat16, packed=False), self._model(mx.bfloat16)
        self._generate(packed, self._inputs(mx.bfloat16))
        for model in (original, packed):
            if conversion == "storage":
                model.set_dtype(mx.float16)
            else:
                model.apply_compute_precision(ComputePrecision(mx.float16))
        dtype = mx.float16 if conversion == "storage" else mx.bfloat16
        inputs = self._inputs(dtype)
        actual = self._generate(packed, inputs)[-1]
        expected = self._generate(original, inputs)[-1]
        self._assert_close(actual, expected, 2e-2)
        assert actual.dtype == expected.dtype
        if conversion == "compute":
            with pytest.raises(ValueError, match="Cannot save"):
                ComputePrecision.ensure_savable(packed)

    @pytest.mark.parametrize("cached", [False, True])
    def test_reference_attention_preserves_prefix_and_target_outputs(self, cached):
        original, packed = self._model(packed=False), self._model()
        latents, text = self._inputs()
        latents = mx.concatenate([latents, latents + 0.25], axis=1)
        layout = QwenImage21Layout.create(
            mx.array([False, True, False]), [(1, 2, 2)] * 2, self.CONFIG["axes_dims_rope"]
        )
        outputs = []
        for model in (original, packed):
            cache = [] if cached else None
            results = []
            with model.inference_projections():
                for timestep in (0.8, 0.5):
                    result = model.forward_reference(
                        latents.at[:, -4:].add(timestep), text, mx.array([timestep]), layout, cache
                    )
                    mx.eval(result)
                    results.append(result)
            outputs.append(results)
        for actual, expected in zip(outputs[1], outputs[0]):
            self._assert_close(actual, expected)

    def test_parameter_gradients_after_inference_match_unpacked_training(self):
        original, packed = self._model(packed=False), self._model()
        inputs = self._inputs()
        self._generate(packed, inputs)
        hidden = mx.array(np.random.default_rng(28).standard_normal((1, 3, 64)).astype(np.float32))
        gradients = []
        for model in (original, packed):
            mlp = model.transformer_blocks[0].img_mlp
            model.train()
            _, gradient = nn.value_and_grad(mlp, self._loss)(mlp, hidden)
            mx.eval(gradient)
            gradients.append(dict(tree_flatten(gradient)))
        assert gradients[0].keys() == gradients[1].keys()
        for name in gradients[0]:
            self._assert_close(gradients[1][name], gradients[0][name])
        assert float(mx.max(mx.abs(gradients[1]["gate_layer.weight"]))) > 1e-6
        assert float(mx.max(mx.abs(gradients[1]["proj.weight"]))) > 1e-6

    @staticmethod
    def _model(dtype=mx.float32, *, packed=True):
        model = Qwen21Transformer(**TestQwen21PackedProjections.CONFIG)
        rng = np.random.default_rng(741)
        weights = []
        for name, value in sorted(tree_flatten(model.parameters())):
            data = rng.standard_normal(value.shape).astype(np.float32) * 0.1
            if ".norm_q." in name or ".norm_k." in name:
                data += 1
            weights.append((name, mx.array(data).astype(dtype)))
        model.load_weights(weights)
        model.use_packed_projections = packed
        return model

    @staticmethod
    def _inputs(dtype=mx.float32):
        rng = np.random.default_rng(21)
        return (
            mx.array(rng.standard_normal((1, 4, 4)).astype(np.float32)).astype(dtype),
            mx.array(rng.standard_normal((1, 3, 16)).astype(np.float32)).astype(dtype),
        )

    @staticmethod
    def _generate(model, inputs):
        with model.inference_projections():
            return [TestQwen21PackedProjections._step(model, inputs, timestep) for timestep in (1, 2)]

    @staticmethod
    def _step(model, inputs, timestep):
        config = Config(
            width=32,
            height=32,
            guidance=1.0,
            scheduler="linear",
            model_config=ModelConfig.qwen_image_21(),
            num_inference_steps=40,
        )
        result = model(timestep, config, inputs[0], inputs[1])
        mx.eval(result)
        return result

    @staticmethod
    def _adapter(linear):
        adapter = LoRALinear.from_linear(linear, r=2)
        rng = np.random.default_rng(77)
        adapter.lora_A = mx.array(rng.standard_normal(adapter.lora_A.shape).astype(np.float32) * 0.2)
        adapter.lora_B = mx.array(rng.standard_normal(adapter.lora_B.shape).astype(np.float32) * 0.2)
        return adapter

    @staticmethod
    def _loss(model, hidden):
        return mx.mean(mx.square(model(hidden)))

    @staticmethod
    def _numpy(array):
        return np.asarray(array.astype(mx.float32))

    @staticmethod
    def _assert_close(actual, expected, tolerance=2e-5):
        np.testing.assert_allclose(
            TestQwen21PackedProjections._numpy(actual),
            TestQwen21PackedProjections._numpy(expected),
            atol=tolerance,
            rtol=tolerance,
        )
