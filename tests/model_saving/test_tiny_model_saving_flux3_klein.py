import pytest

from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from mflux.models.flux3.model.flux3_transformer.flux3_transformer import Flux3Transformer
from mflux.models.flux3.weights.flux3_weight_definition import (
    Flux3KleinBaseWeightDefinition,
    Flux3KleinWeightDefinition,
)
from tests.model_saving.tiny_checkpoint_helper import TinyCheckpointRoundtrip, TinyVAEStandIn


class TestTinyFlux3KleinModelSaving:
    # There is no slow save/load twin for FLUX 3 Image Klein; this is its checkpoint coverage. The same
    # ModelSaver -> WeightLoader -> WeightApplier seam as the other tiny tests, on real Flux3Transformer and
    # Qwen3TextEncoder classes at tiny dimensions.
    @pytest.mark.fast
    @pytest.mark.parametrize(
        ("weight_definition", "flow_map"),
        [(Flux3KleinWeightDefinition, True), (Flux3KleinBaseWeightDefinition, False)],
        ids=["distilled", "base"],
    )
    def test_tiny_quantized_checkpoint_roundtrips_exactly(self, tmp_path, weight_definition, flow_map):
        TinyCheckpointRoundtrip.save_and_reload_expecting_identical_weights(
            weight_definition=weight_definition,
            make_components=lambda: TestTinyFlux3KleinModelSaving._tiny_components(flow_map),
            base_path=tmp_path / "flux3_klein_tiny_q8",
            bits=8,
            # Force shard boundaries so the index.json / weight_map multi-shard paths are exercised.
            tensors_per_shard=8,
        )

    @staticmethod
    def _tiny_components(flow_map: bool):
        # inner_dim = heads * head_dim = 64, the smallest multiple of the quantization group size; axes_dims_rope
        # must sum to attention_head_dim.
        return {
            "vae": TinyVAEStandIn(),
            "transformer": Flux3Transformer(
                in_channels=128,
                cond_in_channels=128,
                num_layers=2,
                num_single_layers=2,
                attention_head_dim=32,
                num_attention_heads=2,
                joint_attention_dim=128,
                axes_dims_rope=(8, 8, 8, 8),
                flow_map=flow_map,
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
