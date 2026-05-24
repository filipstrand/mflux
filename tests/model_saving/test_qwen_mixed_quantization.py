from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition


class QuantizableModule:
    def to_quantized(self):
        return self


def test_qwen_q4_keeps_conditioning_paths_unquantized():
    module = QuantizableModule()

    skipped_paths = [
        "img_in",
        "txt_in",
        "time_text_embed.timestep_embedder.linear_1",
        "time_text_embed.timestep_embedder.linear_2",
        "transformer_blocks.0.img_mod_linear",
        "transformer_blocks.0.txt_mod_linear",
        "norm_out.linear",
        "proj_out",
    ]

    for path in skipped_paths:
        assert QwenWeightDefinition.quantization_predicate(path, module, 4) is False


def test_qwen_q4_quantizes_bulk_transformer_paths():
    module = QuantizableModule()

    quantized_paths = [
        "transformer_blocks.0.attn.to_q",
        "transformer_blocks.0.attn.add_v_proj",
        "transformer_blocks.0.img_ff.mlp_in",
        "transformer_blocks.0.txt_ff.mlp_out",
    ]

    for path in quantized_paths:
        assert QwenWeightDefinition.quantization_predicate(path, module, 4) is True


def test_qwen_q8_quantizes_conditioning_paths():
    module = QuantizableModule()

    assert QwenWeightDefinition.quantization_predicate("img_in", module, 8) is True
    assert QwenWeightDefinition.quantization_predicate("transformer_blocks.0.img_mod_linear", module, 8) is True


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


def test_qwen_q4_loaded_mixed_layout_uses_mixed_quantization_predicate():
    module = QuantizableModule()
    weights = LoadedWeights(
        components={"transformer": {"img_in": {"weight": object(), "bias": object()}}},
        meta_data=MetaData(quantization_level=4),
    )

    predicate = QwenWeightDefinition.quantization_predicate_for_loaded_weights(weights=weights, bits=4)

    assert predicate("img_in", module, 4) is False
