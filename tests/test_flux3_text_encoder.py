"""FLUX 3 prompt encoding: reference positions, text ids, and the optional position_ids of the shared Qwen3 encoder."""

import mlx.core as mx
import numpy as np
import pytest

from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from mflux.models.flux3.model.flux3_text_encoder import Flux3PromptEncoder
from mflux.models.flux3.weights.flux3_weight_definition import Flux3KleinWeightDefinition


@pytest.mark.fast
class TestFlux3TextEncoder:
    def test_positions_are_cumsum_with_padding_set_to_one(self):
        mask = mx.array([[1, 1, 1, 0, 0]])
        assert Flux3PromptEncoder.positions(mask).tolist() == [[0, 1, 2, 1, 1]]

    def test_text_ids_use_a_plain_range_not_cumsum_positions(self):
        ids = np.array(Flux3PromptEncoder.prepare_text_ids(4))
        assert ids.dtype == np.float32
        assert ids.tolist() == [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3]]

    def test_text_encoder_key_transform_keeps_language_layers_up_to_32(self):
        transform = Flux3KleinWeightDefinition._text_encoder_key
        assert transform("model.language_model.layers.31.mlp.up_proj.weight") == "model.layers.31.mlp.up_proj.weight"
        assert transform("model.language_model.layers.32.mlp.up_proj.weight") is None
        assert transform("model.language_model.embed_tokens.weight") == "model.embed_tokens.weight"
        assert transform("model.visual.blocks.0.attn.qkv.weight") is None

    def test_default_positions_match_explicit_range_and_padding_positions_matter(self):
        mx.random.seed(0)
        encoder = Qwen3TextEncoder(
            vocab_size=64,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            head_dim=16,
            rope_theta=5e6,
        )
        ids = mx.array([[5, 6, 7, 0, 0]])
        mask = mx.array([[1, 1, 1, 0, 0]])
        default, _ = encoder(ids, mask)
        explicit, _ = encoder(ids, mask, position_ids=mx.arange(5)[None])
        assert mx.array_equal(default, explicit)
        custom, _ = encoder(ids, mask, position_ids=Flux3PromptEncoder.positions(mask))
        assert mx.allclose(default[:, :3], custom[:, :3])  # real tokens never see padded positions
        assert not mx.allclose(default[:, 3:], custom[:, 3:])
