from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition


class QuantizableModule:
    def __init__(self, weight_shape=(64, 64)):
        self.weight = Tensor(weight_shape)

    def to_quantized(self):
        return self


class Tensor:
    def __init__(self, shape):
        self.shape = shape


def test_qwen_q4_uses_q8_image_modulation():
    module = QuantizableModule()

    assert QwenWeightDefinition.quantization_predicate("transformer_blocks.0.img_mod_linear", module, 4) == {"bits": 8}


def test_qwen_q4_quantizes_bulk_and_small_transformer_paths():
    module = QuantizableModule()

    quantized_paths = [
        "img_in",
        "txt_in",
        "time_text_embed.timestep_embedder.linear_1",
        "time_text_embed.timestep_embedder.linear_2",
        "norm_out.linear",
        "proj_out",
        "transformer_blocks.0.attn.to_q",
        "transformer_blocks.0.attn.add_v_proj",
        "transformer_blocks.0.img_ff.mlp_in",
        "transformer_blocks.0.txt_ff.mlp_out",
        "transformer_blocks.0.txt_mod_linear",
    ]

    for path in quantized_paths:
        assert QwenWeightDefinition.quantization_predicate(path, module, 4) is True


def test_qwen_q8_quantizes_conditioning_paths():
    module = QuantizableModule()

    assert QwenWeightDefinition.quantization_predicate("img_in", module, 8) is True
    assert QwenWeightDefinition.quantization_predicate("transformer_blocks.0.img_mod_linear", module, 8) is True


def test_qwen_q4_quantizes_text_encoder_language_to_q4_and_vision_to_q8():
    module = QuantizableModule()

    assert QwenWeightDefinition.quantization_predicate("encoder.layers.0.self_attn.q_proj", module, 4) is True
    assert QwenWeightDefinition.quantization_predicate("encoder.embed_tokens", module, 4) is True
    assert QwenWeightDefinition.quantization_predicate("encoder.visual.blocks.0.attn.qkv", module, 4) == {"bits": 8}


def test_qwen_q4_skips_text_encoder_layers_that_are_not_group64_compatible():
    module = QuantizableModule(weight_shape=(1280, 3420))

    assert QwenWeightDefinition.quantization_predicate("encoder.visual.blocks.0.mlp.down_proj", module, 4) is False


def test_qwen_q8_quantizes_text_encoder_to_q8():
    module = QuantizableModule()

    assert QwenWeightDefinition.quantization_predicate("encoder.layers.0.self_attn.q_proj", module, 8) is True
    assert QwenWeightDefinition.quantization_predicate("encoder.visual.blocks.0.attn.qkv", module, 8) is True


def test_weight_applier_passes_bits_to_bit_aware_predicates():
    calls = []

    def predicate(path, module, bits):
        calls.append((path, module, bits))
        return bits == 4

    module = QuantizableModule()
    wrapped = WeightApplier.quantization_predicate_for_bits(predicate, 4)

    assert wrapped("path", module) is True
    assert calls == [("path", module, 4)]


def test_qwen_q4_loaded_legacy_layout_keeps_legacy_quantization_predicate():
    module = QuantizableModule()
    weights = LoadedWeights(
        components={"transformer": {"img_in": {"weight": object(), "scales": object()}}},
        meta_data=MetaData(quantization_level=4),
    )

    predicate = QwenWeightDefinition.quantization_predicate_for_loaded_weights(weights=weights, bits=4)

    assert predicate("img_in", module, 4) is True


def test_qwen_q4_loaded_img_mod_q8_layout_uses_mixed_quantization_predicate():
    module = QuantizableModule()
    weights = LoadedWeights(
        components={
            "transformer": {
                "img_in": {"weight": Tensor((3072, 8)), "scales": Tensor((3072, 1))},
                "transformer_blocks": [
                    {
                        "img_mod_linear": {
                            "weight": Tensor((18432, 768)),
                            "scales": Tensor((18432, 48)),
                        },
                        "txt_mod_linear": {
                            "weight": Tensor((18432, 384)),
                            "scales": Tensor((18432, 48)),
                        },
                    }
                ],
            }
        },
        meta_data=MetaData(quantization_level=4),
    )

    predicate = QwenWeightDefinition.quantization_predicate_for_loaded_weights(weights=weights, bits=4)

    assert predicate("img_in", module, 4) is True
    assert predicate("transformer_blocks.0.img_mod_linear", module, 4) == {"bits": 8}
    assert predicate("transformer_blocks.0.txt_mod_linear", module, 4) is True


def test_qwen_loaded_bf16_text_encoder_layout_skips_text_encoder_quantization():
    module = QuantizableModule()
    weights = LoadedWeights(
        components={
            "transformer": {
                "img_in": {"weight": Tensor((3072, 8)), "scales": Tensor((3072, 1))},
                "transformer_blocks": [
                    {
                        "img_mod_linear": {
                            "weight": Tensor((18432, 768)),
                            "scales": Tensor((18432, 48)),
                        },
                    }
                ],
            },
            "text_encoder": {
                "encoder": {
                    "layers": [
                        {
                            "self_attn": {
                                "q_proj": {"weight": object()},
                            },
                        }
                    ],
                }
            },
        },
        meta_data=MetaData(quantization_level=4),
    )

    predicate = QwenWeightDefinition.quantization_predicate_for_loaded_weights(weights=weights, bits=4)

    assert predicate("encoder.layers.0.self_attn.q_proj", module, 4) is False


def test_qwen_loaded_mixed_text_encoder_layout_uses_saved_text_encoder_bits():
    module = QuantizableModule()
    incompatible_module = QuantizableModule(weight_shape=(1280, 3420))
    weights = LoadedWeights(
        components={
            "transformer": {
                "img_in": {"weight": Tensor((3072, 8)), "scales": Tensor((3072, 1))},
                "transformer_blocks": [
                    {
                        "img_mod_linear": {
                            "weight": Tensor((18432, 768)),
                            "scales": Tensor((18432, 48)),
                        },
                    }
                ],
            },
            "text_encoder": {
                "encoder": {
                    "layers": [
                        {
                            "self_attn": {
                                "q_proj": {
                                    "weight": Tensor((3584, 448)),
                                    "scales": Tensor((3584, 56)),
                                },
                            },
                        }
                    ],
                    "visual": {
                        "blocks": [
                            {
                                "attn": {
                                    "qkv": {
                                        "weight": Tensor((3840, 320)),
                                        "scales": Tensor((3840, 20)),
                                    },
                                },
                            }
                        ],
                    },
                }
            },
        },
        meta_data=MetaData(quantization_level=4),
    )

    predicate = QwenWeightDefinition.quantization_predicate_for_loaded_weights(weights=weights, bits=4)

    assert predicate("encoder.layers.0.self_attn.q_proj", module, 4) is True
    assert predicate("encoder.visual.blocks.0.attn.qkv", module, 4) == {"bits": 8}
    assert predicate("encoder.visual.blocks.0.mlp.down_proj", incompatible_module, 4) is False


def test_qwen_q4_loaded_bf16_img_mod_layout_uses_bf16_img_mod_predicate():
    module = QuantizableModule()
    weights = LoadedWeights(
        components={
            "transformer": {
                "img_in": {"weight": object(), "bias": object()},
                "transformer_blocks": [
                    {
                        "img_mod_linear": {"weight": object(), "bias": object()},
                        "txt_mod_linear": {"weight": object(), "scales": object()},
                    }
                ],
            }
        },
        meta_data=MetaData(quantization_level=4),
    )

    predicate = QwenWeightDefinition.quantization_predicate_for_loaded_weights(weights=weights, bits=4)

    assert predicate("img_in", module, 4) is False
    assert predicate("transformer_blocks.0.img_mod_linear", module, 4) is False
    assert predicate("transformer_blocks.0.txt_mod_linear", module, 4) is True


def test_qwen_q4_loaded_post1_bf16_mixed_layout_keeps_txt_mod_unquantized():
    module = QuantizableModule()
    weights = LoadedWeights(
        components={
            "transformer": {
                "img_in": {"weight": object(), "bias": object()},
                "transformer_blocks": [
                    {
                        "img_mod_linear": {"weight": object(), "bias": object()},
                        "txt_mod_linear": {"weight": object(), "bias": object()},
                    }
                ],
            }
        },
        meta_data=MetaData(quantization_level=4),
    )

    predicate = QwenWeightDefinition.quantization_predicate_for_loaded_weights(weights=weights, bits=4)

    assert predicate("img_in", module, 4) is False
    assert predicate("transformer_blocks.0.img_mod_linear", module, 4) is False
    assert predicate("transformer_blocks.0.txt_mod_linear", module, 4) is False
    assert predicate("transformer_blocks.0.attn.to_q", module, 4) is True
