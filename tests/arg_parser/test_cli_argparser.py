import json
from pathlib import Path
from unittest.mock import patch

import pytest

from mflux.ui.cli.parsers import CommandLineParser


def _create_mflux_generate_parser(with_controlnet=False) -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image based on a prompt.")
    parser.add_model_arguments(require_model_arg=False)
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_lora_arguments()
    parser.add_image_to_image_arguments(required=False)
    if with_controlnet:
        parser.add_controlnet_arguments()
    parser.add_output_arguments()
    return parser


@pytest.fixture
def mflux_generate_parser() -> CommandLineParser:
    return _create_mflux_generate_parser(with_controlnet=False)


@pytest.fixture
def mflux_generate_controlnet_parser() -> CommandLineParser:
    return _create_mflux_generate_parser(with_controlnet=True)


@pytest.fixture
def mflux_save_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Save a quantized version of Flux.1 to disk.")  # fmt: off
    parser.add_model_arguments(path_type="save", require_model_arg=True)
    parser.add_lora_arguments()
    return parser


@pytest.fixture
def mflux_generate_minimal_argv() -> list[str]:
    return ["mflux-generate", "--prompt", "meaning of life"]


@pytest.fixture
def mflux_generate_controlnet_minimal_argv() -> list[str]:
    return ["mflux-generate-controlnet", "--prompt", "meaning of life, imitated"]


@pytest.fixture
def temp_dir(tmp_path_factory) -> Path:
    # Create a temporary directory for the module
    temp_dir = tmp_path_factory.mktemp("mflux_cli_argparser_tests")
    return Path(temp_dir)


@pytest.fixture
def base_metadata_dict() -> dict:
    return {
        "mflux_version": "0.4.0",
        "model": "dev",
        "seed": 42042,
        "steps": 14,
        "guidance": None,
        "precision": "mlx.core.bfloat16",
        "quantize": None,
        "generation_time_seconds": 42.0,
        "lora_paths": None,
        "lora_scales": None,
        "init_image": None,
        "init_image_strength": None,
        "controlnet_image": None,
        "controlnet_strength": None,
        "controlnet_save_canny": False,
    }


def test_model_path_requires_model_arg(mflux_generate_parser):
    # when loading a model via --path, the model name still need to be specified
    with patch("sys.argv", "mflux-generate", "--path", "/some/saved/model"):
        assert pytest.raises(SystemExit, mflux_generate_parser.parse_args)


def test_model_arg_not_in_file(mflux_generate_parser, mflux_generate_minimal_argv, base_metadata_dict, temp_dir):
    metadata_file = temp_dir / "model.json"
    with metadata_file.open("wt") as m:
        del base_metadata_dict["model"]
        json.dump(base_metadata_dict, m, indent=4)
    # test model arg not provided in either flag or file
    with patch('sys.argv', mflux_generate_minimal_argv + ['--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        pytest.raises(SystemExit, mflux_generate_parser.parse_args)
    # test value read from flag
    with patch('sys.argv', mflux_generate_minimal_argv + ['--model', 'dev', '--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.model == "dev"
    # test value read from flag
    with patch('sys.argv', mflux_generate_minimal_argv + ['--model', 'schnell', '--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.model == "schnell"


def test_model_arg_in_file(mflux_generate_parser, mflux_generate_minimal_argv, base_metadata_dict, temp_dir):
    metadata_file = temp_dir / "model.json"
    with metadata_file.open("wt") as m:
        base_metadata_dict["model"] = "dev"
        json.dump(base_metadata_dict, m, indent=4)
    # test value read from file
    with patch('sys.argv', mflux_generate_minimal_argv + ['--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.model == "dev"
    # test value read from flag, overrides value from file
    with patch('sys.argv', mflux_generate_minimal_argv + ['--model', 'schnell', '--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.model == "schnell"


def test_prompt_arg(mflux_generate_parser, mflux_generate_minimal_argv, base_metadata_dict, temp_dir):
    metadata_file = temp_dir / "prompt.json"
    file_prompt = "origin of the universe"
    with metadata_file.open("wt") as m:
        base_metadata_dict["prompt"] = file_prompt
        json.dump(base_metadata_dict, m, indent=4)
    # test metadata config accepted, use mflux_generate_minimal_argv without fixture --prompt
    with patch('sys.argv', mflux_generate_minimal_argv[:-2] + ['--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.prompt == file_prompt
    # test CLI override, use mflux_generate_minimal_argv without fixture --prompt
    cli_prompt = "place where monsters come from"
    with patch('sys.argv', mflux_generate_minimal_argv[:-2] + ['--prompt', cli_prompt, '--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.prompt == cli_prompt


def test_guidance_arg(mflux_generate_parser, mflux_generate_minimal_argv, base_metadata_dict, temp_dir):  # fmt: off
    metadata_file = temp_dir / "guidance.json"
    with metadata_file.open("wt") as m:
        base_metadata_dict["guidance"] = 4.2
        json.dump(base_metadata_dict, m, indent=4)
    # test metadata config accepted
    with patch('sys.argv', mflux_generate_minimal_argv + ['--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.guidance == pytest.approx(4.2)
    # test CLI override
    with patch('sys.argv', mflux_generate_minimal_argv + ['--guidance', '5.0', '--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.guidance == pytest.approx(5.0)


def test_quantize_arg(mflux_generate_parser, mflux_generate_minimal_argv, base_metadata_dict, temp_dir):  # fmt: off
    metadata_file = temp_dir / "quantize.json"
    with metadata_file.open("wt") as m:
        base_metadata_dict["quantize"] = 4
        json.dump(base_metadata_dict, m, indent=4)
    # test metadata config accepted
    with patch('sys.argv', mflux_generate_minimal_argv + ['--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.quantize == 4
    # test CLI override
    with patch('sys.argv', mflux_generate_minimal_argv + ['--quantize', '8', '--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.quantize == 8


def test_seed_arg(mflux_generate_parser, mflux_generate_minimal_argv, base_metadata_dict, temp_dir):  # fmt: off
    metadata_file = temp_dir / "seed.json"
    with metadata_file.open("wt") as m:
        base_metadata_dict["seed"] = 24
        json.dump(base_metadata_dict, m, indent=4)
    # test metadata config accepted
    with patch('sys.argv', mflux_generate_minimal_argv + ['--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.seed == 24
    # test CLI override
    with patch('sys.argv', mflux_generate_minimal_argv + ['--seed', '2424', '--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.seed == 2424


def test_steps_arg(mflux_generate_parser, mflux_generate_minimal_argv, base_metadata_dict, temp_dir):  # fmt: off
    metadata_file = temp_dir / "steps.json"
    with metadata_file.open("wt") as m:
        base_metadata_dict["steps"] = 8
        json.dump(base_metadata_dict, m, indent=4)

    # test user default value for dev
    with patch("sys.argv", mflux_generate_minimal_argv + ["--model", "dev"]):
        args = mflux_generate_parser.parse_args()
        assert args.steps == 14

    # test user default value for schnell
    with patch("sys.argv", mflux_generate_minimal_argv + ["--model", "schnell"]):
        args = mflux_generate_parser.parse_args()
        assert args.steps == 4

    # test metadata config accepted
    with patch('sys.argv', mflux_generate_minimal_argv + ['--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.steps == 8

    # test CLI override
    with patch('sys.argv', mflux_generate_minimal_argv + ['--steps', '12', '--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.steps == 12


def test_lora_args(mflux_generate_parser, mflux_generate_minimal_argv, base_metadata_dict, temp_dir):  # fmt: off
    test_paths = ["/some/lora/1.safetensors", "/some/lora/2.safetensors"]
    metadata_file = temp_dir / "lora_args.json"
    with metadata_file.open("wt") as m:
        base_metadata_dict["lora_paths"] = test_paths
        base_metadata_dict["lora_scales"] = [0.3, 0.7]
        json.dump(base_metadata_dict, m, indent=4)

    # test user default value
    with patch("sys.argv", mflux_generate_minimal_argv + ["-m", "schnell"]):
        args = mflux_generate_parser.parse_args()
        assert args.lora_paths is None
        assert args.lora_scales is None

    # test metadata config accepted
    with patch('sys.argv', mflux_generate_minimal_argv + ['--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.lora_paths == test_paths
        assert args.lora_scales == [pytest.approx(0.3), pytest.approx(0.7)]

    # test CLI override that merges CLI loras and config file loras
    new_loras = ["--lora-paths", "/some/lora/3.safetensors", "/some/lora/4.safetensors", "--lora-scales", "0.1", "0.9"]
    with patch('sys.argv', mflux_generate_minimal_argv + new_loras + ['--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert len(args.lora_paths) == 4
        assert args.lora_paths == test_paths + new_loras[1:3]
        assert len(args.lora_scales) == 4
        assert args.lora_scales == [pytest.approx(v) for v in [0.3, 0.7, 0.1, 0.9]]


def test_image_to_image_args(mflux_generate_parser, mflux_generate_minimal_argv, base_metadata_dict, temp_dir):  # fmt: off
    metadata_file = temp_dir / "image_to_image.json"
    test_path = "/some/awesome/image.png"
    with metadata_file.open("wt") as m:
        base_metadata_dict["init_image_path"] = test_path
        json.dump(base_metadata_dict, m, indent=4)

    # test user default value
    with patch("sys.argv", mflux_generate_minimal_argv + ["-m", "dev"]):
        args = mflux_generate_parser.parse_args()
        assert args.init_image_path is None
        assert args.init_image_strength == 0.4  # default

    # test metadata config accepted
    with patch('sys.argv', mflux_generate_minimal_argv + ['--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.init_image_path == test_path
        assert args.init_image_strength == 0.4  # default

    # test strength override
    with patch('sys.argv', mflux_generate_minimal_argv + ['--init-image-strength', '0.7', '--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.init_image_path == test_path
        assert args.init_image_strength == 0.7

    # test image path override
    with patch('sys.argv', mflux_generate_minimal_argv + ['--init-image-path', '/some/better/image.png', '--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.init_image_path == Path("/some/better/image.png")
        assert args.init_image_strength == 0.4  # default


def test_controlnet_args(mflux_generate_controlnet_parser, mflux_generate_controlnet_minimal_argv, base_metadata_dict, temp_dir):  # fmt: off
    test_path = "/some/cnet/1.safetensors"
    metadata_file = temp_dir / "cnet_args.json"
    with metadata_file.open("wt") as m:
        base_metadata_dict["controlnet_image_path"] = test_path
        base_metadata_dict["controlnet_strength"] = 0.48
        json.dump(base_metadata_dict, m, indent=4)

    # test metadata config accepted
    with patch('sys.argv', mflux_generate_controlnet_minimal_argv + ['--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_controlnet_parser.parse_args()
        assert args.controlnet_image_path == test_path
        assert args.controlnet_strength == pytest.approx(0.48)
        assert args.controlnet_save_canny is False

    # test CLI override
    override_cnet = [
        "--controlnet-image-path",
        "/some/lora/2.safetensors",
        "--controlnet-strength",
        "0.85",
        "--controlnet-save-canny",
    ]
    with patch('sys.argv', mflux_generate_controlnet_minimal_argv + override_cnet + ['--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_controlnet_parser.parse_args()
        assert args.controlnet_image_path == "/some/lora/2.safetensors"
        assert args.controlnet_strength == pytest.approx(0.85)
        assert args.controlnet_save_canny is True

    # test controlnet_save_canny is False when not specified
    with metadata_file.open("wt") as m:
        del base_metadata_dict["controlnet_save_canny"]
        json.dump(base_metadata_dict, m, indent=4)

    with patch('sys.argv', mflux_generate_controlnet_minimal_argv + ['--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_controlnet_parser.parse_args()
        assert args.controlnet_save_canny is False


def test_save_args(mflux_save_parser):
    with patch("sys.argv", ["mflux-save", "--model", "dev"]):
        # required --path not provided, exits to error
        assert pytest.raises(SystemExit, mflux_save_parser.parse_args)
    with patch("sys.argv", ["mflux-save", "--model", "dev", "--path", "/some/model/folder"]):
        # required --path not provided, exits to error
        args = mflux_save_parser.parse_args()
        assert args.path == "/some/model/folder"
