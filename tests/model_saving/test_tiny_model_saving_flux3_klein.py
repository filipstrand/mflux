import re
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mflux.models.common.config import ModelConfig
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.common.weights.mapping.weight_mapper import WeightMapper
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from mflux.models.flux3.flux3_initializer import Flux3Initializer
from mflux.models.flux3.model.flux3_transformer.flux3_transformer import Flux3Transformer
from mflux.models.flux3.weights.flux3_weight_definition import (
    Flux3KleinBaseWeightDefinition,
    Flux3KleinWeightDefinition,
)
from mflux.utils.exceptions import ModelConfigError
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

    # A saved MFlux model is guessed as Distilled unless --base-model says otherwise. Loading a checkpoint with the
    # other kind's definition must fail before the models are built, naming the --base-model that fits.
    @pytest.mark.fast
    @pytest.mark.parametrize(
        ("saved_flow_map", "config_name", "expected"),
        [
            (False, "flux3-image-klein", "--base-model flux3-image-klein-base"),
            (True, "flux3-image-klein-base", "--base-model flux3-image-klein"),
        ],
        ids=["base_read_as_distilled", "distilled_read_as_base"],
    )
    def test_checkpoint_kind_mismatch_names_the_right_base_model(self, tmp_path, saved_flow_map, config_name, expected):
        weights, model_config = TestTinyFlux3KleinModelSaving._save_and_load(tmp_path, saved_flow_map, config_name)

        with pytest.raises(ModelConfigError, match=re.escape(expected) + r"\."):
            Flux3Initializer._check_checkpoint_kind(weights, model_config, str(tmp_path))

    @pytest.mark.fast
    @pytest.mark.parametrize(
        ("saved_flow_map", "config_name"),
        [(True, "flux3-image-klein"), (False, "flux3-image-klein-base")],
        ids=["distilled", "base"],
    )
    def test_checkpoint_kind_match_passes(self, tmp_path, saved_flow_map, config_name):
        weights, model_config = TestTinyFlux3KleinModelSaving._save_and_load(tmp_path, saved_flow_map, config_name)

        Flux3Initializer._check_checkpoint_kind(weights, model_config, str(tmp_path))

    @staticmethod
    def _save_and_load(base_path, saved_flow_map: bool, config_name: str):
        saved_definition = Flux3KleinWeightDefinition if saved_flow_map else Flux3KleinBaseWeightDefinition
        components = TestTinyFlux3KleinModelSaving._tiny_components(saved_flow_map)
        TinyCheckpointRoundtrip._quantize(components, saved_definition, 8)
        ModelSaver.save_model(
            model=TinyCheckpointRoundtrip._as_model(components, saved_definition),
            bits=8,
            base_path=str(base_path),
            weight_definition=saved_definition,
        )
        model_config = ModelConfig.from_name(config_name)
        definition = Flux3Initializer.weight_definition(model_config)
        return WeightLoader.load(weight_definition=definition, model_path=str(base_path)), model_config


@pytest.mark.fast
@pytest.mark.parametrize(
    ("hf_keys", "raises"),
    [
        (
            ["proj_out.weight", "target_timestep_embedder.linear_1.weight", "target_timestep_embedder.linear_2.weight"],
            True,
        ),
        (["proj_out.weight"], False),
    ],
    ids=["distilled_hub_layout_read_as_base", "base_hub_layout"],
)
def test_base_mapping_keeps_the_target_time_embedding_for_the_kind_check(hf_keys, raises):
    # The diffusers-layout repos go through the mapping before the check: the Base mapping must keep the embedder.
    hf_weights = {key: mx.zeros((4, 4)) for key in hf_keys}
    transformer = next(c for c in Flux3KleinBaseWeightDefinition.get_components() if c.name == "transformer")
    mapped = WeightMapper.apply_mapping(hf_weights, transformer.mapping_getter())
    weights = SimpleNamespace(transformer=mapped)
    model_config = ModelConfig.from_name("flux3-image-klein-base")
    if raises:
        with pytest.raises(ModelConfigError, match=re.escape("--base-model flux3-image-klein") + r"\."):
            Flux3Initializer._check_checkpoint_kind(weights, model_config, "black-forest-labs/flux-3-image-klein")
    else:
        Flux3Initializer._check_checkpoint_kind(weights, model_config, "black-forest-labs/flux-3-image-klein-base")
