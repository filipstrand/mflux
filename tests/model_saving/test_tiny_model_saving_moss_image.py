import pytest

from mflux.models.common_models.qwen3_5 import Qwen35TextModel
from mflux.models.moss_image.model.moss_image_transformer.moss_image_transformer import MOSSImageTransformer
from mflux.models.moss_image.weights.moss_image_weight_definition import MOSSImageWeightDefinition
from tests.model_saving.tiny_checkpoint_helper import TinyCheckpointRoundtrip, TinyVAEStandIn


class TestTinyMOSSImageModelSaving:
    @pytest.mark.fast
    def test_tiny_quantized_checkpoint_roundtrips_exactly(self, tmp_path):
        # MOSS-Image has no slow save test. This runs the ModelSaver -> WeightLoader ->
        # WeightApplier seam with the real MOSS-Image component classes at tiny dimensions.
        TinyCheckpointRoundtrip.save_and_reload_expecting_identical_weights(
            weight_definition=MOSSImageWeightDefinition,
            make_components=TestTinyMOSSImageModelSaving._tiny_components,
            base_path=tmp_path / "moss_image_tiny_q8",
            bits=8,
            # Force shard boundaries so index.json/weight_map multi-shard paths are
            # exercised — the size-based split never shards test-sized tensors.
            tensors_per_shard=8,
        )

    @staticmethod
    def _tiny_components():
        # Every input dimension is a multiple of 64 (MLX's quantization group size).
        # The transformer needs head_dim (128 / 2 heads) divisible by 8. Four text-encoder
        # layers give three Gated DeltaNet layers and one full-attention layer.
        return {
            "vae": TinyVAEStandIn(),
            "transformer": MOSSImageTransformer(
                dim=128,
                ffn_dim=128,
                num_heads=2,
                num_layers=2,
                text_dim=128,
                freq_dim=64,
                latent_channels=128,
            ),
            "text_encoder": Qwen35TextModel(
                vocab_size=128,
                hidden_size=128,
                intermediate_size=128,
                num_hidden_layers=4,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=64,
                linear_num_key_heads=2,
                linear_num_value_heads=2,
                linear_key_head_dim=64,
                linear_value_head_dim=64,
            ),
        }
