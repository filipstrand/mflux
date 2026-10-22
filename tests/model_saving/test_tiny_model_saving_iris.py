import mlx.core as mx
import pytest

from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from mflux.models.iris.model.iris_transformer.iris_transformer import IrisTransformer
from mflux.models.iris.weights.iris_weight_definition import IrisTurboWeightDefinition, IrisWeightDefinition
from tests.model_saving.tiny_checkpoint_helper import TinyCheckpointRoundtrip


class TestTinyIrisModelSaving:
    # There is no slow save/load twin for Iris-3B; this is its checkpoint coverage. The same
    # ModelSaver -> WeightLoader -> WeightApplier seam as the other tiny tests, on real IrisTransformer and
    # Qwen3TextEncoder classes at tiny dimensions. Turbo and base save the same layout.
    @pytest.mark.fast
    @pytest.mark.parametrize(
        "weight_definition", [IrisTurboWeightDefinition, IrisWeightDefinition], ids=["turbo", "base"]
    )
    def test_tiny_quantized_checkpoint_roundtrips_exactly(self, tmp_path, weight_definition):
        TinyCheckpointRoundtrip.save_and_reload_expecting_identical_weights(
            weight_definition=weight_definition,
            make_components=TestTinyIrisModelSaving._tiny_components,
            base_path=tmp_path / "iris_tiny_q8",
            bits=8,
            # Force shard boundaries so the index.json / weight_map multi-shard paths are exercised.
            tensors_per_shard=16,
        )

    @pytest.mark.fast
    def test_the_saved_layout_is_transformer_and_text_encoder_directories(self, tmp_path):
        from mflux.models.common.weights.saving.model_saver import ModelSaver

        components = TestTinyIrisModelSaving._tiny_components()
        ModelSaver.save_model(
            model=TinyCheckpointRoundtrip._as_model(components, IrisWeightDefinition),
            bits=None,
            base_path=str(tmp_path),
            weight_definition=IrisWeightDefinition,
        )

        assert sorted(path.name for path in tmp_path.iterdir()) == ["text_encoder", "transformer"]

    @pytest.mark.fast
    def test_the_tiny_transformer_runs_a_forward_pass(self):
        transformer = TestTinyIrisModelSaving._tiny_components()["transformer"]

        out = transformer(mx.zeros((1, 3, 16, 24)), mx.array([500.0]), mx.zeros((1, 16, 3, 128)), mx.ones((1, 16)))

        assert out.shape == (1, 3, 16, 24)

    @staticmethod
    def _tiny_components():
        # Every width is a multiple of the quantization group size (64) except the layer pool and the pixel stage,
        # which the predicate leaves dense; head_dim stays a multiple of 4 for the 2D rotary table.
        return {
            "transformer": IrisTransformer(
                hidden_size=128,
                depth=3,
                dual_depth=1,
                num_heads=2,
                num_kv_heads=1,
                patch_size=4,
                in_channels=3,
                text_len=16,
                text_dim=128,
                text_num_layers=3,
                text_layer_num_heads=2,
                pixel_depth=2,
                pixel_hidden_size=16,
                pixel_attn_hidden_size=128,
                pixel_num_heads=2,
            ),
            "text_encoder": Qwen3TextEncoder(
                vocab_size=128,
                hidden_size=128,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=64,
                intermediate_size=128,
            ),
        }
