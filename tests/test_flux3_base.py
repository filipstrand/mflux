"""FLUX 3 Image Klein Base (undistilled, CFG): registry entry, weight mapping, CFG-free-of-target transformer."""

import mlx.core as mx
import pytest

from mflux.cli.defaults import defaults as ui_defaults
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.resolution.model_revisions import ModelRevisions
from mflux.models.flux3.flux3_initializer import Flux3Initializer
from mflux.models.flux3.model.flux3_transformer.flux3_transformer import Flux3Transformer
from mflux.models.flux3.weights.flux3_weight_definition import (
    Flux3KleinBaseWeightDefinition,
    Flux3KleinWeightDefinition,
)
from mflux.models.flux3.weights.flux3_weight_mapping import Flux3WeightMapping


@pytest.mark.fast
class TestFlux3Base:
    def test_registry_entry(self):
        base = ModelConfig.flux3_image_klein_base()
        assert base.supports_guidance and base.transformer_overrides == {"flow_map": False}
        assert not ModelConfig.flux3_image_klein().supports_guidance
        for alias in ("flux3-base", "flux3-klein-base", "flux3-image-klein-base"):
            assert ModelConfig.from_name(alias) is base
        assert ui_defaults.model_inference_steps("flux3-base") == 50
        assert ui_defaults.model_inference_steps("flux3") == 16
        assert ModelRevisions.get(base.model_name) == "1469d27cfb9bb1bd81d3a93f5a023b39dbf136e4"

    def test_weight_definition_selected_by_flow_map(self):
        assert Flux3Initializer.weight_definition(ModelConfig.flux3_image_klein()) is Flux3KleinWeightDefinition
        assert (
            Flux3Initializer.weight_definition(ModelConfig.flux3_image_klein_base()) is Flux3KleinBaseWeightDefinition
        )

    def test_base_mapping_maps_the_target_timestep_embedder_only_optionally(self):
        # Base has no target_timestep_embedder; mapping it optionally lets the initializer detect a Distilled
        # checkpoint loaded as Base.
        required = lambda targets: {t.to_pattern for t in targets if t.required}  # noqa: E731
        distilled = required(Flux3WeightMapping.get_transformer_mapping())
        base_targets = Flux3WeightMapping.get_base_transformer_mapping()
        assert distilled - required(base_targets) == {
            "target_timestep_embedder.linear_1.weight",
            "target_timestep_embedder.linear_2.weight",
        }
        optional = {t.to_pattern for t in base_targets if not t.required}
        assert optional == {"target_timestep_embedder.linear_1.weight", "target_timestep_embedder.linear_2.weight"}
        components = {c.name: c for c in Flux3KleinBaseWeightDefinition.get_components()}
        assert components["transformer"].mapping_getter is Flux3WeightMapping.get_base_transformer_mapping
        assert components["vae"].mapping_getter is Flux3KleinWeightDefinition.get_components()[0].mapping_getter

    def test_transformer_target_timestep_only_for_flow_map(self):
        small = dict(num_layers=1, num_single_layers=1, num_attention_heads=1, attention_head_dim=128, joint_attention_dim=16)  # fmt: skip
        base, distilled = Flux3Transformer(flow_map=False, **small), Flux3Transformer(flow_map=True, **small)
        assert base.target_timestep_embedder is None and distilled.target_timestep_embedder is not None
        args = dict(
            hidden_states=mx.zeros((1, 4, 128)),
            img_ids=mx.zeros((4, 4)),
            encoder_hidden_states=mx.zeros((1, 3, 16)),
            txt_ids=mx.zeros((3, 4)),
            timestep=mx.array([0.5]),
        )
        assert base(**args).shape == (1, 4, 128)
        with pytest.raises(ValueError, match="target_timestep"):
            base(**args, target_timestep=mx.array([0.0]))
        with pytest.raises(ValueError, match="target_timestep"):
            distilled(**args)
