from pathlib import Path

import pytest
from mlx.utils import tree_flatten

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.iris.iris_initializer import IrisInitializer
from mflux.models.iris.model.iris_transformer.iris_transformer import IrisTransformer
from mflux.models.iris.weights.iris_weight_definition import IrisTurboWeightDefinition, IrisWeightDefinition
from mflux.models.iris.weights.iris_weight_mapping import IrisWeightMapping

CHECKPOINT_TENSORS = 639


@pytest.mark.fast
class TestIrisWeightMapping:
    def test_every_module_parameter_is_loaded_exactly_once_and_nothing_else(self):
        targets = IrisWeightMapping.get_transformer_mapping()
        parameters = {name for name, _ in tree_flatten(IrisTransformer().parameters())}

        destinations = [target.to_pattern for target in targets]
        assert len(destinations) == len(set(destinations)) == CHECKPOINT_TENSORS
        assert set(destinations) == parameters

    def test_every_checkpoint_name_is_read_once(self):
        sources = [name for target in IrisWeightMapping.get_transformer_mapping() for name in target.from_pattern]

        assert len(sources) == len(set(sources)) == CHECKPOINT_TENSORS

    def test_nn_sequential_indices_are_renamed(self):
        pairs = {t.to_pattern: t.from_pattern[0] for t in IrisWeightMapping.get_transformer_mapping()}

        assert pairs["t_embedder.linear_1.weight"] == "t_embedder.mlp.0.weight"
        assert pairs["t_embedder.linear_2.bias"] == "t_embedder.mlp.2.bias"
        assert pairs["y_embedder.layer_blocks.1.mlp.fc2.weight"] == "y_embedder.layer_blocks.1.mlp.2.weight"

    def test_the_first_eight_blocks_are_dual_stream_and_the_rest_single_stream(self):
        names = {t.to_pattern for t in IrisWeightMapping.get_transformer_mapping()}

        assert "blocks.7.attn.q_proj_x.weight" in names
        assert "blocks.8.attn.q_proj_x.weight" not in names
        assert "blocks.8.q_proj.weight" in names
        assert "blocks.23.q_proj.weight" in names
        assert "blocks.24.q_proj.weight" not in names


@pytest.mark.fast
class TestIrisQuantizationPredicate:
    @staticmethod
    def _quantized_paths(bits: int | None = 8) -> set[str]:
        model = IrisTransformer()
        return {
            path
            for path, module in model.named_modules()
            if IrisWeightDefinition.quantization_predicate(path, module, bits)
        }

    def test_large_linears_quantize(self):
        paths = TestIrisQuantizationPredicate._quantized_paths()

        for expected in (
            "s_embedder.proj",
            "t_embedder.linear_1",
            "modulation_cores.adaln_img",
            "blocks.0.attn.q_proj_x",
            "blocks.0.mlp_x.w1",
            "blocks.9.q_proj",
            "blocks.9.mlp.w1",
            "y_embedder.layer_blocks.0.attn.qkv",
            "y_embedder.layer_blocks.0.mlp.fc2",
            "y_embedder.refiner.proj",
            "pixel_blocks.0.adaln",
            "pixel_blocks.0.compress",
            "pixel_blocks.0.expand",
            "pixel_blocks.0.attn.qkv",
        ):
            assert expected in paths, expected

    def test_the_per_pixel_adaln_projections_stay_dense_at_4_bits_and_below_only(self):
        # At q4 they print a screen-door grain over flat areas; q8 is clean, so it keeps them quantized.
        adaln = {f"pixel_blocks.{i}.adaln" for i in range(4)}

        assert adaln <= TestIrisQuantizationPredicate._quantized_paths(8)
        assert adaln <= TestIrisQuantizationPredicate._quantized_paths(None)
        for bits in (4, 3):
            paths = TestIrisQuantizationPredicate._quantized_paths(bits)
            assert not adaln & paths
            assert {"pixel_blocks.0.compress", "pixel_blocks.0.expand", "pixel_blocks.0.attn.qkv"} <= paths

    def test_the_predicate_takes_the_bit_width_the_applier_passes_through(self):
        from mflux.models.common.weights.loading.weight_applier import WeightApplier

        wrapped = WeightApplier._predicate_with_bits(IrisWeightDefinition.quantization_predicate, 4)
        model = IrisTransformer()
        modules = dict(model.named_modules())

        assert not wrapped("pixel_blocks.1.adaln", modules["pixel_blocks.1.adaln"])
        assert wrapped("pixel_blocks.1.compress", modules["pixel_blocks.1.compress"])

    def test_swiglu_output_projections_stay_dense_because_6826_is_not_a_multiple_of_the_group_size(self):
        paths = TestIrisQuantizationPredicate._quantized_paths()

        assert not [p for p in paths if p.endswith(".w2")]

    def test_the_small_pixel_stage_layers_and_the_layer_pool_stay_dense(self):
        paths = TestIrisQuantizationPredicate._quantized_paths()

        for excluded in ("pixel_embedder.proj", "final_layer.linear", "y_embedder.layer_pool"):
            assert excluded not in paths
        assert not [p for p in paths if p.startswith("pixel_blocks") and ".mlp." in p]

    def test_no_quantized_layer_has_an_input_width_off_the_group_size(self):
        model = IrisTransformer()
        for path, module in model.named_modules():
            if IrisWeightDefinition.quantization_predicate(path, module):
                assert module.weight.shape[-1] % 64 == 0, path


@pytest.mark.fast
class TestIrisWeightDefinition:
    def test_saved_layout_lists_transformer_text_encoder_and_tokenizer(self):
        components = IrisWeightDefinition.get_components()

        assert [(c.name, c.hf_subdir) for c in components] == [
            ("transformer", "transformer"),
            ("text_encoder", "text_encoder"),
        ]
        assert [(t.name, t.hf_subdir) for t in IrisWeightDefinition.get_tokenizers()] == [("qwen3_vl", "tokenizer")]

    def test_turbo_and_base_read_the_same_repo_from_different_places(self):
        turbo, base = IrisTurboWeightDefinition.source_transformer(), IrisWeightDefinition.source_transformer()

        assert (turbo.hf_subdir, base.hf_subdir) == ("turbo", "")
        assert turbo.weight_files == base.weight_files == ["model.safetensors"]
        assert IrisTurboWeightDefinition.SOURCE_TRANSFORMER_PATTERN == "turbo/model.safetensors"
        assert IrisWeightDefinition.SOURCE_TRANSFORMER_PATTERN == "model.safetensors"

    def test_the_text_encoder_keeps_all_36_layers_and_drops_the_vision_tower(self):
        component = IrisWeightDefinition.source_text_encoder()

        assert component.num_layers == 36
        assert component.key_transform("model.language_model.layers.35.self_attn.q_proj.weight") == (
            "model.layers.35.self_attn.q_proj.weight"
        )
        assert component.key_transform("model.visual.blocks.0.attn.qkv.weight") is None
        assert component.key_transform("lm_head.weight") is None

    def test_the_variant_comes_from_the_model_config(self):
        assert IrisInitializer.weight_definition(ModelConfig.iris_3b()) is IrisTurboWeightDefinition
        assert IrisInitializer.weight_definition(ModelConfig.iris_3b_base()) is IrisWeightDefinition

    def test_both_entries_share_the_pinned_repo(self):
        from mflux.models.common.resolution.model_revisions import ModelRevisions

        assert ModelConfig.iris_3b_base().model_name == ModelConfig.iris_3b().model_name == "speridlabs/iris-3b"
        assert ModelRevisions.get("speridlabs/iris-3b")
        assert ModelRevisions.get(IrisWeightDefinition.TEXT_ENCODER_REPO)


@pytest.mark.fast
class TestIrisWeightSources:
    @pytest.fixture
    def loader_calls(self, monkeypatch):
        calls = {"single": [], "load": []}

        def load_single(component, repo_id, file_pattern="*.safetensors"):
            calls["single"].append((component.name, repo_id, file_pattern))
            return LoadedWeights(components={component.name: {}}, meta_data=_meta())

        def load(weight_definition, model_path=None, download_patterns=None):
            calls["load"].append((weight_definition, model_path))
            return LoadedWeights(components={"transformer": {}, "text_encoder": {}}, meta_data=_meta())

        monkeypatch.setattr("mflux.models.iris.iris_initializer.WeightLoader.load_single", staticmethod(load_single))
        monkeypatch.setattr("mflux.models.iris.iris_initializer.WeightLoader.load", staticmethod(load))
        return calls

    @pytest.mark.parametrize("model_path", [None, "speridlabs/iris-3b"], ids=["no_path", "repo_id_from_a_sidecar"])
    def test_without_a_saved_model_the_two_upstream_repos_are_read(self, loader_calls, model_path):
        config = ModelConfig.iris_3b()
        saved_path = None if model_path in (None, config.model_name) else model_path
        IrisInitializer._load_weights(IrisTurboWeightDefinition, config, saved_path)

        assert loader_calls["load"] == []
        assert loader_calls["single"] == [
            ("transformer", "speridlabs/iris-3b", "turbo/model.safetensors"),
            ("text_encoder", "Qwen/Qwen3-VL-4B-Instruct", "model-*.safetensors"),
        ]

    def test_a_saved_model_is_read_from_its_own_directory_only(self, loader_calls):
        IrisInitializer._load_weights(IrisTurboWeightDefinition, ModelConfig.iris_3b(), "/models/iris-q8")

        assert loader_calls["single"] == []
        assert loader_calls["load"] == [(IrisTurboWeightDefinition, "/models/iris-q8")]

    def test_init_treats_the_repo_id_as_no_saved_model(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(IrisInitializer, "_load_weights", staticmethod(lambda d, c, p: seen.setdefault("path", p)))
        monkeypatch.setattr(
            IrisInitializer, "_init_tokenizers", staticmethod(lambda m, d, p: seen.setdefault("tok", p))
        )
        monkeypatch.setattr(IrisInitializer, "_init_models", staticmethod(lambda m: None))
        monkeypatch.setattr(IrisInitializer, "_apply_weights", staticmethod(lambda *a: None))

        IrisInitializer.init(
            model=type("M", (), {})(),
            model_config=ModelConfig.iris_3b_base(),
            quantize=None,
            model_path="speridlabs/iris-3b",
        )

        assert seen == {"path": None, "tok": None}


def _meta():
    from mflux.models.common.weights.loading.loaded_weights import MetaData

    return MetaData()


@pytest.mark.fast
def test_the_checkpoint_fixture_lists_639_tensors():
    names = (Path(__file__).parent / "resources" / "checkpoint_keys" / "iris_3b_transformer.txt").read_text().split()

    assert len(names) == CHECKPOINT_TENSORS
