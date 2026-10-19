import json
import sys
import warnings
from argparse import Namespace

import mlx.core as mx
import pytest

from mflux.callbacks.callback_manager import CallbackManager
from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.callbacks.instances.memory_saver import MemorySaver
from mflux.cli.capabilities import describe_command
from mflux.cli.defaults import defaults as ui_defaults
from mflux.models.common.cli import save
from mflux.models.common.config.model_config import AVAILABLE_MODELS, ModelConfig
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.iris.cli import iris_generate
from mflux.models.iris.iris_initializer import IrisInitializer
from mflux.models.iris.latent_creator.iris_latent_creator import IrisLatentCreator
from mflux.models.iris.variants import Iris


class _StubIris:
    # Records what the CLI hands the model instead of loading 14 GB of weights.
    instances: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.generated: list = []
        self.callbacks = CallbackRegistry()
        _StubIris.instances.append(self)

    def generate_image(self, **kwargs):
        self.generated.append(kwargs)
        return type("Image", (), {"save": staticmethod(lambda path, export_json_metadata=False: None)})()


@pytest.fixture
def run_cli(monkeypatch):
    def run(*argv):
        _StubIris.instances.clear()
        monkeypatch.setattr(sys, "argv", ["mflux-generate-iris", "--prompt", "a fox", "--seed", "1", *argv])
        monkeypatch.setattr(iris_generate, "Iris", _StubIris)
        monkeypatch.setattr(iris_generate.CallbackManager, "register_callbacks", staticmethod(lambda **kw: None))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            iris_generate.main()
        return _StubIris.instances[0], [str(w.message) for w in caught]

    return run


@pytest.mark.fast
class TestIrisGenerateCli:
    def test_default_model_is_turbo_at_1024_with_four_steps(self, run_cli):
        model, warned = run_cli()

        assert model.kwargs["model_config"] is AVAILABLE_MODELS["iris-3b"]
        assert model.generated[0]["width"] == model.generated[0]["height"] == 1024
        assert model.generated[0]["num_inference_steps"] == 4
        assert warned == []

    def test_the_base_model_is_selected_with_model_iris_3b_base_and_defaults_to_100_steps(self, run_cli):
        model, warned = run_cli("--model", "iris-3b-base")

        assert model.kwargs["model_config"] is AVAILABLE_MODELS["iris-3b-base"]
        assert model.generated[0]["num_inference_steps"] == 100
        assert model.generated[0]["guidance"] is None
        assert warned == []

    def test_turbo_warns_about_options_it_ignores(self, run_cli):
        _, warned = run_cli("--steps", "9", "--guidance", "5", "--negative-prompt", "blur", "--scheduler", "linear")

        for option in ("--steps", "--guidance", "--negative-prompt", "--scheduler"):
            assert any(message.startswith(f"{option} is ignored") for message in warned), option

    def test_base_honours_steps_guidance_and_negative_prompt(self, run_cli):
        model, warned = run_cli(
            "--model", "iris-3b-base", "--steps", "9", "--guidance", "5", "--negative-prompt", "blur"
        )

        assert model.generated[0]["num_inference_steps"] == 9
        assert model.generated[0]["guidance"] == 5.0
        assert model.generated[0]["negative_prompt"] == "blur"
        assert warned == []

    def test_vae_options_are_ignored_because_there_is_no_vae(self, run_cli):
        _, warned = run_cli("--vae-tiling", "--vae-tile-size", "256")

        assert any(message.startswith("--vae-tiling is ignored") for message in warned)
        assert any(message.startswith("--vae-tile-size is ignored") for message in warned)

    @pytest.mark.parametrize("size", [["--width", "1000"], ["--height", "1030"], ["--width", "500", "--height", "500"]])
    def test_sizes_that_are_not_multiples_of_16_exit_before_the_model_loads(self, run_cli, capsys, size):
        with pytest.raises(SystemExit) as exit_info:
            run_cli(*size)

        assert exit_info.value.code == 2
        assert "multiples of 16" in capsys.readouterr().err
        assert _StubIris.instances == []

    def test_multiples_of_16_pass_through(self, run_cli):
        model, _ = run_cli("--width", "512", "--height", "768")

        assert (model.generated[0]["width"], model.generated[0]["height"]) == (512, 768)

    def test_quantize_and_a_saved_model_path_reach_the_model(self, run_cli):
        model, _ = run_cli("--model", "/models/iris-3b-turbo-q8", "-q", "8")

        assert model.kwargs["model_path"] == "/models/iris-3b-turbo-q8"
        assert model.kwargs["quantize"] == 8
        assert model.kwargs["model_config"] is AVAILABLE_MODELS["iris-3b"]

    def test_a_saved_base_model_loads_as_base_by_name_or_by_base_model_flag(self, run_cli):
        by_name, _ = run_cli("--model", "/models/iris-3b-base-q4")
        by_flag, _ = run_cli("--model", "/models/my-q4", "--base-model", "iris-3b-base")

        assert by_name.kwargs["model_config"] is AVAILABLE_MODELS["iris-3b-base"]
        assert by_flag.kwargs["model_config"] is AVAILABLE_MODELS["iris-3b-base"]

    def test_a_foreign_model_is_rejected(self, run_cli):
        with pytest.raises(Exception, match="only accepts the aliases"):
            run_cli("--model", "dev")

    def test_every_alias_resolves_to_its_own_entry(self):
        for key, aliases in {"iris-3b": ["iris-3b-turbo", "iris-turbo", "iris"], "iris-3b-base": ["iris-base"]}.items():
            for alias in [key, *aliases]:
                assert ConfigResolution.resolve(alias) is AVAILABLE_MODELS[key], alias

    def test_capabilities_report_the_conditional_and_ignored_options(self):
        entry = describe_command("mflux-generate-iris", iris_generate.__name__)
        status = {option["flag"]: option["status"] for option in entry["options"]}

        assert status["--steps"] == status["--guidance"] == status["--negative-prompt"] == "conditional"
        assert status["--scheduler"] == status["--vae-tiling"] == status["--vae-tile-size"] == "ignored"
        assert "--compute-precision" not in status
        assert "--lora" not in status


@pytest.mark.fast
class TestIrisPathInference:
    # A saved model is told apart by its directory name, resolved by the same call the CLI makes: the bare name is
    # the fast default (as for FLUX.2 and FLUX 3) and only a "base" in the name selects the undistilled model.
    @pytest.mark.parametrize(
        ("directory", "expected"),
        [
            ("~/models/iris-3b-q8", "iris-3b"),
            ("~/models/iris-q8", "iris-3b"),
            ("~/models/my-iris", "iris-3b"),
            ("~/models/iris-3b-turbo-q8", "iris-3b"),
            ("~/models/my-finetune", "iris-3b"),
            ("~/models/iris-3b-base-q8", "iris-3b-base"),
            ("~/models/iris-base-q8", "iris-3b-base"),
        ],
    )
    def test_a_saved_directory_resolves_to_the_entry_its_name_selects(self, directory, expected):
        config = ConfigResolution.resolve_restricted(
            None,
            iris_generate.DEFAULT_MODEL,
            model_path=directory,
            extra_keys=iris_generate.FAMILY_MODELS,
        )

        assert config is AVAILABLE_MODELS[expected]


@pytest.mark.fast
class TestIrisDefaults:
    def test_default_steps(self):
        assert ui_defaults.MODEL_INFERENCE_STEPS["iris-3b"] == 4
        assert ui_defaults.MODEL_INFERENCE_STEPS["iris-3b-base"] == 100

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("/models/iris-3b-turbo-q8", "iris-3b"),
            ("/models/iris-3b-q8", "iris-3b"),
            ("/models/iris-3b-base-q8", "iris-3b-base"),
            ("/models/iris-base-q4", "iris-3b-base"),
            ("/models/iris-q4", "iris-3b"),
        ],
    )
    def test_a_saved_model_steps_like_the_entry_its_name_selects(self, name, expected):
        assert ui_defaults.model_inference_steps(name) == ui_defaults.MODEL_INFERENCE_STEPS[expected]

    def test_the_registry_entries(self):
        turbo, base = ModelConfig.iris_3b(), ModelConfig.iris_3b_base()

        assert (turbo.supports_guidance, base.supports_guidance) == (False, True)
        assert turbo.max_sequence_length == base.max_sequence_length == 300
        assert turbo.text_encoder_overrides == base.text_encoder_overrides
        assert turbo.text_encoder_overrides["num_hidden_layers"] == 36
        assert turbo.text_encoder_overrides["rope_theta"] == 5_000_000.0
        assert turbo.priority != base.priority


@pytest.mark.fast
class TestIrisSidecarReplay:
    @staticmethod
    def _model(key: str) -> Iris:
        model = Iris.__new__(Iris)
        model.model_config = AVAILABLE_MODELS[key]
        return model

    def test_turbo_records_its_own_entry(self):
        model_config = TestIrisSidecarReplay._model("iris-3b")._metadata_model_config()

        assert model_config.base_model is None

    def test_base_records_itself_as_the_base_model_because_both_entries_share_one_repo(self):
        base = TestIrisSidecarReplay._model("iris-3b-base")
        model_config = base._metadata_model_config()

        assert model_config.model_name == "speridlabs/iris-3b"
        assert model_config.base_model == "iris-3b-base"
        assert AVAILABLE_MODELS["iris-3b-base"].base_model is None

    @pytest.mark.parametrize(("base_model", "expected"), [(None, "iris-3b"), ("iris-3b-base", "iris-3b-base")])
    def test_a_sidecar_replays_on_the_entry_that_wrote_it(self, monkeypatch, tmp_path, base_model, expected):
        sidecar = tmp_path / "image.metadata.json"
        sidecar.write_text(
            json.dumps(
                {"model": "speridlabs/iris-3b", "base_model": base_model, "prompt": "a fox", "seed": 3, "steps": 4}
            )
        )
        monkeypatch.setattr(sys, "argv", ["prog", "--config-from-conf", str(sidecar)])
        args = iris_generate.build_parser().parse_args()

        config = ConfigResolution.resolve_restricted(
            args.model,
            iris_generate.DEFAULT_MODEL,
            model_path=args.model_path,
            extra_keys=iris_generate.FAMILY_MODELS,
            base_model=args.base_model,
        )

        assert config is AVAILABLE_MODELS[expected]


@pytest.mark.fast
class TestIrisSaveDispatch:
    @pytest.mark.parametrize(
        ("name", "key"),
        [
            ("iris-3b", "iris-3b"),
            ("iris-turbo", "iris-3b"),
            ("iris", "iris-3b"),
            ("iris-3b-base", "iris-3b-base"),
            ("iris-base", "iris-3b-base"),
        ],
    )
    def test_every_name_routes_to_the_iris_class(self, name, key):
        parser = save.CommandLineParser(description="x")

        resolved = save.SaveDispatch.resolve_registry_key(parser, name, None)

        assert resolved == key
        assert save.MODEL_CLASSES[resolved] is Iris

    @pytest.mark.parametrize("path", ["/models/iris-3b-q8", "/models/iris-3b-base-q4", "/models/my-iris"])
    def test_a_saved_directory_loads_from_itself_when_mflux_save_resolves_it(self, path):
        # mflux-save builds a config whose model_name is the path itself; that path is still a saved model.
        config = ConfigResolution.resolve(path, base_model="iris-3b")

        assert config.model_name == path
        assert IrisInitializer.saved_path(config, path) == path
        # Iris(model_config=config) without model_path loads the same saved directory.
        assert IrisInitializer.saved_path(config, None) == path

    @pytest.mark.parametrize("key", ["iris-3b", "iris-3b-base"])
    def test_the_upstream_repo_id_is_not_a_saved_model(self, key):
        config = AVAILABLE_MODELS[key]

        assert IrisInitializer.saved_path(config, None) is None
        assert IrisInitializer.saved_path(config, config.model_name) is None


@pytest.mark.fast
class TestIrisLowRam:
    @staticmethod
    def _model() -> Iris:
        # What the initializer leaves behind: a text encoder, a prompt cache and no VAE of any kind.
        model = Iris.__new__(Iris)
        model.callbacks = CallbackRegistry()
        model.tiling_config = None
        model.text_encoder = object()
        model.prompt_cache = {"a fox": (mx.zeros((1,)), mx.zeros((1,)))}
        model.transformer = object()
        return model

    @staticmethod
    def _args(**overrides) -> Namespace:
        values = dict(
            low_ram=True, mlx_cache_limit_gb=None, vae_tiling=False, vae_tile_size=None, seed=[1], image_path=[]
        )
        return Namespace(**{**values, **overrides})

    def test_low_ram_registers_without_touching_a_vae_and_evicts_the_text_encoder_after_encoding(self):
        model = TestIrisLowRam._model()
        assert not hasattr(model, "vae")

        saver = CallbackManager._register_memory_saver(TestIrisLowRam._args(), model)
        assert isinstance(saver, MemorySaver)
        saver.call_before_loop(seed=1, prompt="a fox", latents=mx.zeros((1,)), config=None)

        assert model.text_encoder is None
        assert not hasattr(model, "vae")

    def test_low_ram_releases_the_transformer_after_the_loop(self):
        model = TestIrisLowRam._model()
        saver = CallbackManager._register_memory_saver(TestIrisLowRam._args(), model)

        saver.call_after_loop(seed=1, prompt="a fox", latents=mx.zeros((1,)), config=None)

        assert model.transformer is None

    def test_several_seeds_keep_the_transformer(self):
        model = TestIrisLowRam._model()
        saver = CallbackManager._register_memory_saver(TestIrisLowRam._args(seed=[1, 2]), model)

        saver.call_after_loop(seed=1, prompt="a fox", latents=mx.zeros((1,)), config=None)

        assert model.transformer is not None


@pytest.mark.fast
class TestIrisModel:
    def test_sizes_must_be_multiples_of_16(self):
        model = Iris.__new__(Iris)
        model.model_config = AVAILABLE_MODELS["iris-3b"]

        with pytest.raises(ValueError, match="multiples of 16"):
            model.generate_image(seed=1, prompt="x", width=1000, height=1024)

    def test_turbo_step_callbacks_get_the_state_as_latents_and_the_prediction_as_denoised(self, monkeypatch):
        # A model that always predicts the same image: every clean prediction is that image, while Turbo's next
        # state is noised again. Callbacks get that state as latents and the prediction as denoised.
        image = mx.full((1, 3, 32, 32), 0.25)
        monkeypatch.setattr(IrisLatentCreator, "create_noise", staticmethod(lambda seed, h, w: mx.ones((1, 3, h, w))))
        monkeypatch.setattr(Iris, "_predict", lambda self, x, model_time, cond, mask: (x - image) / (model_time / 1000))
        seen = []

        class Recorder:
            def call_in_loop(self, t, seed, prompt, latents, config, time_steps, denoised=None):
                seen.append((latents, denoised))

        model = Iris.__new__(Iris)
        model.model_config = AVAILABLE_MODELS["iris-3b"]
        model.callbacks = CallbackRegistry()
        model.callbacks.register(Recorder())
        model.prompt_cache = {"x": (mx.zeros((1,)), mx.zeros((1,)))}
        model.bits = None

        model.generate_image(seed=1, prompt="x", width=32, height=32)

        assert len(seen) == 4
        assert all(mx.allclose(denoised, image, atol=1e-5).item() for _, denoised in seen)
        # The steps before the last pass the noised state. The state after the last step is the image.
        assert not any(mx.allclose(latents, image, atol=1e-5).item() for latents, _ in seen[:-1])
        assert mx.allclose(seen[-1][0], image, atol=1e-5).item()

    def test_turbo_is_a_property_of_the_config(self):
        turbo, base = Iris.__new__(Iris), Iris.__new__(Iris)
        turbo.model_config, base.model_config = AVAILABLE_MODELS["iris-3b"], AVAILABLE_MODELS["iris-3b-base"]

        assert (turbo.is_turbo, base.is_turbo) == (True, False)

    def test_the_noise_is_seeded_float32_in_pixel_space(self):
        noise = IrisLatentCreator.create_noise(seed=7, height=64, width=96)

        assert noise.shape == (1, 3, 64, 96)
        assert noise.dtype == mx.float32
        assert mx.array_equal(noise, IrisLatentCreator.create_noise(seed=7, height=64, width=96)).item()
        assert not mx.array_equal(noise, IrisLatentCreator.create_noise(seed=8, height=64, width=96)).item()

    def test_unpack_latents_is_the_identity(self):
        latents = mx.ones((1, 3, 32, 32))

        assert IrisLatentCreator.unpack_latents(latents=latents, height=32, width=32) is latents
