import json
import random
from pathlib import Path
from unittest.mock import patch

import pytest

from mflux.ui import defaults as ui_defaults
from mflux.ui.box_values import BoxValues
from mflux.ui.cli.parsers import CommandLineParser


def _create_mflux_generate_parser(with_controlnet=False, require_model_arg=False) -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image based on a prompt.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=require_model_arg)
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_lora_arguments()
    parser.add_image_to_image_arguments(required=False)
    parser.add_image_outpaint_arguments()
    if with_controlnet:
        parser.add_controlnet_arguments(mode="canny")
    parser.add_output_arguments()
    return parser


@pytest.fixture
def mflux_generate_parser() -> CommandLineParser:
    return _create_mflux_generate_parser(with_controlnet=False, require_model_arg=False)


@pytest.fixture
def mflux_generate_controlnet_parser() -> CommandLineParser:
    return _create_mflux_generate_parser(with_controlnet=True, require_model_arg=False)


@pytest.fixture
def mflux_save_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Save a quantized version of Flux.1 to disk.")  # fmt: off
    parser.add_general_arguments()
    parser.add_model_arguments(path_type="save", require_model_arg=True)
    parser.add_lora_arguments()
    return parser


@pytest.fixture
def mflux_generate_minimal_argv() -> list[str]:
    return ["mflux-generate", "--prompt", "meaning of life"]


@pytest.fixture
def mflux_generate_minimal_model_argv() -> list[str]:
    return ["mflux-generate", "--prompt", "meaning of life", "--model", "dev"]


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
        "image": None,
        "image_strength": None,
        "controlnet_image": None,
        "controlnet_strength": None,
        "controlnet_save_canny": False,
    }


@pytest.fixture
def mflux_fill_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image using the fill tool to complete masked areas.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=False)
    parser.add_fill_arguments()
    parser.add_output_arguments()
    return parser


@pytest.fixture
def mflux_fill_minimal_argv() -> list[str]:
    return [
        "mflux-generate-fill",
        "--prompt",
        "meaning of life",
        "--image-path",
        "image.png",
        "--masked-image-path",
        "mask.png",
    ]


@pytest.fixture
def mflux_save_depth_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Save depth map from an image.")
    parser.add_general_arguments()
    parser.add_save_depth_arguments()
    parser.add_output_arguments()
    return parser


@pytest.fixture
def mflux_save_depth_minimal_argv() -> list[str]:
    return ["mflux-save-depth", "--image-path", "image.png"]


@pytest.fixture
def mflux_redux_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate redux images.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_redux_arguments()
    parser.add_output_arguments()
    return parser


@pytest.fixture
def mflux_redux_minimal_argv() -> list[str]:
    return ["mflux-generate-redux", "--redux-image-paths", "image1.png", "image2.png"]


@pytest.fixture
def mflux_concept_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image with concept attention based on a prompt and concept.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_image_to_image_arguments(required=False)
    parser.add_output_arguments()
    parser.add_concept_attention_arguments()
    return parser


@pytest.fixture
def mflux_concept_minimal_argv() -> list[str]:
    return [
        "mflux-concept",
        "--prompt",
        "a beautiful landscape with a car",
        "--concept",
        "car",
    ]


@pytest.fixture
def mflux_catvton_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate virtual try-on images using in-context learning.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=False, require_prompt=False)
    parser.add_catvton_arguments()
    parser.add_in_context_arguments()
    parser.add_output_arguments()
    return parser


@pytest.fixture
def mflux_catvton_minimal_argv() -> list[str]:
    return [
        "mflux-generate-in-context-catvton",
        "--person-image",
        "person.png",
        "--person-mask",
        "person_mask.png",
        "--garment-image",
        "garment.png",
    ]


@pytest.fixture
def mflux_in_context_edit_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate images using in-context editing.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=False, require_prompt=False)
    parser.add_in_context_edit_arguments()
    parser.add_in_context_arguments()
    parser.add_output_arguments()
    return parser


@pytest.fixture
def mflux_in_context_edit_minimal_argv() -> list[str]:
    return [
        "mflux-generate-in-context-edit",
        "--reference-image",
        "reference.png",
        "--instruction",
        "make the hair black",
    ]


def test_model_path_requires_model_arg(mflux_generate_parser):
    # when loading a model via --path, the model name still need to be specified
    with patch("sys.argv", "mflux-generate", "--path", "/some/saved/model"):
        assert pytest.raises(SystemExit, mflux_generate_parser.parse_args)


def test_model_arg_not_in_file(mflux_generate_parser, mflux_generate_minimal_argv, base_metadata_dict, temp_dir):
    metadata_file = temp_dir / "model.json"
    with metadata_file.open("wt") as m:
        del base_metadata_dict["model"]
        json.dump(base_metadata_dict, m, indent=4)

    # Create a parser that requires the model argument
    parser_with_required_model = _create_mflux_generate_parser(require_model_arg=True)

    # test model arg not provided in either flag or file should raise SystemExit with required parser
    with patch('sys.argv', mflux_generate_minimal_argv + ['--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        pytest.raises(SystemExit, parser_with_required_model.parse_args)

    # test value read from flag
    with patch('sys.argv', mflux_generate_minimal_argv + ['--model', 'dev', '--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.model == "dev"
        assert args.base_model is None
    # test value read from flag
    with patch('sys.argv', mflux_generate_minimal_argv + ['--model', 'schnell', '--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.model == "schnell"
        assert args.base_model is None


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


def test_base_model_arg_in_file(mflux_generate_parser, mflux_generate_minimal_argv, base_metadata_dict, temp_dir):
    metadata_file = temp_dir / "model.json"
    with metadata_file.open("wt") as m:
        base_metadata_dict["model"] = "some-lab/some-model"
        base_metadata_dict["base_model"] = "dev"
        json.dump(base_metadata_dict, m, indent=4)
    # test value read from file
    with patch('sys.argv', mflux_generate_minimal_argv + ['--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.model == "some-lab/some-model"
        assert args.base_model == "dev"
    # test value read from flag, overrides value from file
    with patch('sys.argv', mflux_generate_minimal_argv + ['--base-model', 'schnell', '--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.model == "some-lab/some-model"
        # override metadata base model with CLI --base-model
        assert args.base_model == "schnell"


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


def test_prompt_file_arg(mflux_generate_parser, mflux_generate_minimal_argv, temp_dir):
    # Create a prompt file
    prompt_content = "prompt from a file being re-read for each generation"
    prompt_file = temp_dir / "prompt.txt"
    with prompt_file.open("wt") as pf:
        pf.write(prompt_content)

    # Test that prompt-file is correctly read
    with patch('sys.argv', ["mflux-generate", "--prompt-file", prompt_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.prompt_file == prompt_file
        assert args.prompt is None  # prompt should be None since we're using prompt-file


def test_prompt_and_prompt_file_mutually_exclusive(mflux_generate_parser, temp_dir):
    # Create a prompt file
    prompt_file = temp_dir / "prompt.txt"
    with prompt_file.open("wt") as pf:
        pf.write("some prompt content")

    # Test that using both --prompt and --prompt-file raises an error
    with pytest.raises(SystemExit):
        with patch('sys.argv', ["mflux-generate", "--prompt", "direct prompt", "--prompt-file", prompt_file.as_posix()]):  # fmt: off
            mflux_generate_parser.parse_args()


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


def test_seed_arg(mflux_generate_parser, mflux_generate_minimal_model_argv, base_metadata_dict, temp_dir):  # fmt: off
    metadata_file = temp_dir / "seed.json"
    with metadata_file.open("wt") as m:
        base_metadata_dict["seed"] = 24
        json.dump(base_metadata_dict, m, indent=4)
    # test metadata config accepted
    with patch('sys.argv', mflux_generate_minimal_model_argv + ['--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.seed == [24]
        assert "_seed_{seed}" not in args.output

    # test CLI override
    with patch('sys.argv', mflux_generate_minimal_model_argv + ['--seed', '2424', '--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        # --seed arg overrides metadata
        assert args.seed == [2424]
        assert "_seed_{seed}" not in args.output

    with patch('sys.argv', mflux_generate_minimal_model_argv + ['--seed', '2424', '4848', '9696']):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.seed == [2424, 4848, 9696]
        assert "_seed_{seed}" in args.output

    with patch('sys.argv', mflux_generate_minimal_model_argv + ['--auto-seeds', '5', '--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        # auto-seeds defers to value from metadata, and is ignored
        assert len(args.seed) == 1
        assert args.seed == [24]
        assert "_seed_{seed}" not in args.output


def test_auto_seeds_arg(mflux_generate_parser, mflux_generate_minimal_model_argv):
    with patch("sys.argv", mflux_generate_minimal_model_argv + ["--seed", "24", "48", "--auto-seeds", "5"]):
        args = mflux_generate_parser.parse_args()
        # auto-seeds defers to explicit values of --seed
        assert len(args.seed) == 2
        assert args.seed == [24, 48]
        assert "_seed_{seed}" in args.output

    for _ in range(0, 10):
        random_auto_seed_count = random.randint(2, 100)
        with patch("sys.argv", mflux_generate_minimal_model_argv + ["--auto-seeds", str(random_auto_seed_count)]):
            args = mflux_generate_parser.parse_args()
            assert len(set(args.seed)) == random_auto_seed_count
            assert "_seed_{seed}" in args.output
            for _ in args.seed:
                assert isinstance(_, int)


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

    # Mock get_lora_path to bypass file validation for test purposes
    with patch("mflux.ui.cli.parsers.get_lora_path", side_effect=lambda x: x):
        # test metadata config accepted
        with patch('sys.argv', mflux_generate_minimal_argv + ['--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
            args = mflux_generate_parser.parse_args()
            assert args.lora_paths == test_paths
            assert args.lora_scales == [pytest.approx(0.3), pytest.approx(0.7)]

        # test CLI override that merges CLI loras and config file loras
        new_loras = [
            "--lora-paths",
            "/some/lora/3.safetensors",
            "/some/lora/4.safetensors",
            "--lora-scales",
            "0.1",
            "0.9",
        ]
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
        base_metadata_dict["image_path"] = test_path
        json.dump(base_metadata_dict, m, indent=4)

    # test user default value
    with patch("sys.argv", mflux_generate_minimal_argv + ["-m", "dev"]):
        args = mflux_generate_parser.parse_args()
        assert args.image_path is None
        assert args.image_strength == 0.4  # default

    # test metadata config accepted
    with patch('sys.argv', mflux_generate_minimal_argv + ['--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.image_path == test_path
        assert args.image_strength == 0.4  # default

    # test strength override
    with patch('sys.argv', mflux_generate_minimal_argv + ['--image-strength', '0.7', '--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.image_path == test_path
        assert args.image_strength == 0.7

    # test image path override
    with patch('sys.argv', mflux_generate_minimal_argv + ['--image-path', '/some/better/image.png', '--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.image_path == Path("/some/better/image.png")
        assert args.image_strength == 0.4  # default


def test_image_outpaint_args(mflux_generate_parser, mflux_generate_minimal_argv, base_metadata_dict, temp_dir):  # fmt: off
    metadata_file = temp_dir / "image_outpaint.json"
    test_padding = "10,20,30,40"
    with metadata_file.open("wt") as m:
        base_metadata_dict["image_outpaint_padding"] = test_padding
        json.dump(base_metadata_dict, m, indent=4)

    # test user default value
    with patch("sys.argv", mflux_generate_minimal_argv + ["-m", "dev"]):
        args = mflux_generate_parser.parse_args()
        assert args.image_outpaint_padding is None

    # test metadata config accepted
    with patch('sys.argv', mflux_generate_minimal_argv + ['--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.image_outpaint_padding == BoxValues(10, 20, 30, 40)

    # test outpaint padding override in 4-value format
    with patch('sys.argv', mflux_generate_minimal_argv + ['--image-outpaint-padding', '5,15,25,35', '--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.image_outpaint_padding == BoxValues(5, 15, 25, 35)

    # test outpaint padding override in percentages in two-value format
    with patch('sys.argv', mflux_generate_minimal_argv + ['--image-outpaint-padding', '10%,20%', '--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.image_outpaint_padding == BoxValues("10%", "20%", "10%", "20%")

    # test outpaint padding override in percentages in three-value format, mixed int/percentages
    # also allowing whitespace between the box values
    with patch('sys.argv', mflux_generate_minimal_argv + ['--image-outpaint-padding', '10%, 50,   20%', '--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.image_outpaint_padding == BoxValues("10%", 50, "20%", 50)


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


def test_fill_args(mflux_fill_parser, mflux_fill_minimal_argv):
    # Test required arguments
    with patch("sys.argv", mflux_fill_minimal_argv):
        args = mflux_fill_parser.parse_args()
        assert args.prompt == "meaning of life"
        assert args.image_path == Path("image.png")
        assert args.masked_image_path == Path("mask.png")
        # Default guidance for fill should be None (will be set to 30 in generate_fill.py)
        assert args.guidance is None  # Parser doesn't set default, app does

    # Test with missing required arguments
    with patch("sys.argv", ["mflux-fill", "--prompt", "test"]):
        # Missing image_path and masked_image_path should raise SystemExit
        pytest.raises(SystemExit, mflux_fill_parser.parse_args)

    with patch("sys.argv", ["mflux-fill", "--image-path", "image.png", "--masked-image-path", "mask.png"]):
        # Missing prompt should raise SystemExit
        pytest.raises(SystemExit, mflux_fill_parser.parse_args)

    # Test with custom values
    custom_argv = mflux_fill_minimal_argv + ["--guidance", "30", "--steps", "20", "--height", "512", "--width", "512"]
    with patch("sys.argv", custom_argv):
        args = mflux_fill_parser.parse_args()
        assert args.guidance == pytest.approx(30.0)
        assert args.steps == 20
        assert args.height == 512
        assert args.width == 512


def test_fill_args_with_metadata(mflux_fill_parser, mflux_fill_minimal_argv, base_metadata_dict, temp_dir):
    metadata_file = temp_dir / "fill_metadata.json"
    # Set up metadata with fill-related values
    with metadata_file.open("wt") as m:
        # Add masked_image_path to the metadata dictionary
        base_metadata_dict["masked_image_path"] = "metadata_mask.png"
        base_metadata_dict["prompt"] = "from metadata file"
        json.dump(base_metadata_dict, m, indent=4)

    # Test with minimal args and metadata
    # First modify the parser to support metadata config
    mflux_fill_parser.supports_metadata_config = True
    mflux_fill_parser.add_metadata_config()

    # Create a modified version of minimal_argv that includes all required arguments
    # that aren't in metadata
    minimal_metadata_argv = [
        "mflux-fill",
        "--config-from-metadata",
        metadata_file.as_posix(),
        "--prompt",
        "CLI prompt",
        "--image-path",
        "image.png",
        "--masked-image-path",
        "cli_mask.png",
    ]

    # Test command line arguments overriding metadata
    with patch("sys.argv", minimal_metadata_argv):
        args = mflux_fill_parser.parse_args()
        assert args.prompt == "CLI prompt"  # From CLI
        assert args.image_path == Path("image.png")  # From CLI
        assert args.masked_image_path == Path("cli_mask.png")  # From CLI


def test_fill_default_guidance():
    # Create a parser just like in generate_fill.py
    parser = CommandLineParser(description="Generate an image using the fill tool to complete masked areas.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=False)
    parser.add_fill_arguments()
    parser.add_output_arguments()

    # Parse minimal arguments
    with patch(
        "sys.argv", ["mflux-fill", "--prompt", "test", "--image-path", "img.png", "--masked-image-path", "mask.png"]
    ):
        args = parser.parse_args()

        # Verify initial guidance value is None (no default set by parser)
        assert args.guidance is None

        # Simulate what happens in generate_fill.py
        if args.guidance is None:
            args.guidance = ui_defaults.DEFAULT_DEV_FILL_GUIDANCE

        # Now check that guidance is correctly set to the default dev fill guidance
        assert args.guidance == ui_defaults.DEFAULT_DEV_FILL_GUIDANCE


def test_save_depth_args(mflux_save_depth_parser, mflux_save_depth_minimal_argv):
    # Test required arguments
    with patch("sys.argv", mflux_save_depth_minimal_argv):
        args = mflux_save_depth_parser.parse_args()
        assert args.image_path == Path("image.png")
        assert hasattr(args, "quantize")

    # Test with quantized argument
    with patch("sys.argv", mflux_save_depth_minimal_argv + ["--quantize", "4"]):
        args = mflux_save_depth_parser.parse_args()
        assert args.image_path == Path("image.png")
        assert args.quantize == 4

    # Test with output argument
    with patch("sys.argv", mflux_save_depth_minimal_argv + ["--output", "depth_map.png"]):
        args = mflux_save_depth_parser.parse_args()
        assert args.image_path == Path("image.png")
        assert args.output == "depth_map.png"


def test_redux_args(mflux_redux_parser, mflux_redux_minimal_argv):
    # Test required arguments
    with patch("sys.argv", mflux_redux_minimal_argv):
        args = mflux_redux_parser.parse_args()
        assert len(args.redux_image_paths) == 2
        assert args.redux_image_paths[0] == Path("image1.png")
        assert args.redux_image_paths[1] == Path("image2.png")
        assert args.redux_image_strengths is None  # Default should be None

    # Test with more image paths
    with patch("sys.argv", ["mflux-redux", "--redux-image-paths", "image1.png", "image2.png", "image3.png"]):
        args = mflux_redux_parser.parse_args()
        assert len(args.redux_image_paths) == 3
        assert args.redux_image_paths[0] == Path("image1.png")
        assert args.redux_image_paths[1] == Path("image2.png")
        assert args.redux_image_paths[2] == Path("image3.png")

    # Test with redux_image_strengths parameter
    with patch("sys.argv", mflux_redux_minimal_argv + ["--redux-image-strengths", "0.8", "0.5"]):
        args = mflux_redux_parser.parse_args()
        assert len(args.redux_image_paths) == 2
        assert len(args.redux_image_strengths) == 2
        assert args.redux_image_strengths[0] == pytest.approx(0.8)
        assert args.redux_image_strengths[1] == pytest.approx(0.5)

    # Test with single redux_image_strength
    with patch("sys.argv", mflux_redux_minimal_argv + ["--redux-image-strengths", "0.3"]):
        args = mflux_redux_parser.parse_args()
        assert len(args.redux_image_paths) == 2
        assert len(args.redux_image_strengths) == 1
        assert args.redux_image_strengths[0] == pytest.approx(0.3)

    # Test with model argument
    with patch("sys.argv", mflux_redux_minimal_argv + ["--model", "dev"]):
        args = mflux_redux_parser.parse_args()
        assert len(args.redux_image_paths) == 2
        assert args.model == "dev"

    # Test with output argument
    with patch("sys.argv", mflux_redux_minimal_argv + ["--output", "redux_result.png"]):
        args = mflux_redux_parser.parse_args()
        assert len(args.redux_image_paths) == 2
        assert args.output == "redux_result.png"


def test_concept_attention_args(mflux_concept_parser, mflux_concept_minimal_argv):
    # Test required arguments
    with patch("sys.argv", mflux_concept_minimal_argv):
        args = mflux_concept_parser.parse_args()
        assert args.prompt == "a beautiful landscape with a car"
        assert args.concept == "car"
        # Test defaults
        assert args.heatmap_layer_indices == list(range(15, 19))
        assert args.heatmap_timesteps is None

    # Test with missing required concept - should raise SystemExit
    with patch("sys.argv", ["mflux-concept", "--prompt", "test"]):
        pytest.raises(SystemExit, mflux_concept_parser.parse_args)

    # Test with missing regular prompt - should raise SystemExit
    with patch("sys.argv", ["mflux-concept", "--concept", "test concept"]):
        pytest.raises(SystemExit, mflux_concept_parser.parse_args)

    # Test with custom heatmap parameters
    custom_argv = mflux_concept_minimal_argv + [
        "--heatmap-layer-indices",
        "10",
        "11",
        "12",
        "--heatmap-timesteps",
        "0",
        "1",
        "2",
    ]
    with patch("sys.argv", custom_argv):
        args = mflux_concept_parser.parse_args()
        assert args.prompt == "a beautiful landscape with a car"
        assert args.concept == "car"
        assert args.heatmap_layer_indices == [10, 11, 12]
        assert args.heatmap_timesteps == [0, 1, 2]


def test_catvton_args(mflux_catvton_parser, mflux_catvton_minimal_argv):
    # Test required arguments
    with patch("sys.argv", mflux_catvton_minimal_argv):
        args = mflux_catvton_parser.parse_args()
        assert args.person_image == "person.png"
        assert args.person_mask == "person_mask.png"
        assert args.garment_image == "garment.png"

        # Test prompt is None by default (set by the app, not parser)
        assert args.prompt is None

    # Test missing required arguments
    with patch("sys.argv", ["mflux-generate-in-context-catvton"]):
        pytest.raises(SystemExit, mflux_catvton_parser.parse_args)

    with patch("sys.argv", ["mflux-generate-in-context-catvton", "--person-image", "person.png"]):
        pytest.raises(SystemExit, mflux_catvton_parser.parse_args)

    with patch(
        "sys.argv", ["mflux-generate-in-context-catvton", "--person-image", "person.png", "--person-mask", "mask.png"]
    ):
        pytest.raises(SystemExit, mflux_catvton_parser.parse_args)

    # Test custom prompt can be set
    with patch("sys.argv", mflux_catvton_minimal_argv + ["--prompt", "custom prompt"]):
        args = mflux_catvton_parser.parse_args()
        assert args.prompt == "custom prompt"

    # Test VAE tiling split argument
    with patch("sys.argv", mflux_catvton_minimal_argv + ["--vae-tiling"]):
        args = mflux_catvton_parser.parse_args()
        assert args.vae_tiling is True
        assert args.vae_tiling_split == "horizontal"  # Default value from parser


def test_in_context_edit_args(mflux_in_context_edit_parser, mflux_in_context_edit_minimal_argv):
    # Test required arguments with instruction
    with patch("sys.argv", mflux_in_context_edit_minimal_argv):
        args = mflux_in_context_edit_parser.parse_args()
        assert args.reference_image == "reference.png"
        assert args.instruction == "make the hair black"
        assert hasattr(mflux_in_context_edit_parser, "supports_in_context_edit")
        assert mflux_in_context_edit_parser.supports_in_context_edit is True

    # Test with prompt instead of instruction
    with patch(
        "sys.argv",
        [
            "mflux-generate-in-context-edit",
            "--reference-image",
            "reference.png",
            "--prompt",
            "A diptych with custom prompt",
        ],
    ):
        args = mflux_in_context_edit_parser.parse_args()
        assert args.reference_image == "reference.png"
        assert args.prompt == "A diptych with custom prompt"

    # Test missing required arguments
    with patch("sys.argv", ["mflux-generate-in-context-edit"]):
        pytest.raises(SystemExit, mflux_in_context_edit_parser.parse_args)

    with patch("sys.argv", ["mflux-generate-in-context-edit", "--reference-image", "reference.png"]):
        pytest.raises(SystemExit, mflux_in_context_edit_parser.parse_args)

    # Test both prompt and instruction provided (should error)
    with patch(
        "sys.argv",
        [
            "mflux-generate-in-context-edit",
            "--reference-image",
            "reference.png",
            "--prompt",
            "test prompt",
            "--instruction",
            "test instruction",
        ],
    ):
        pytest.raises(SystemExit, mflux_in_context_edit_parser.parse_args)

    # Test VAE tiling split argument
    with patch("sys.argv", mflux_in_context_edit_minimal_argv + ["--vae-tiling"]):
        args = mflux_in_context_edit_parser.parse_args()
        assert args.vae_tiling is True
        assert args.vae_tiling_split == "horizontal"  # Default value from parser


def test_in_context_args(mflux_catvton_parser, mflux_catvton_minimal_argv):
    # Test save_full_image flag
    with patch("sys.argv", mflux_catvton_minimal_argv):
        args = mflux_catvton_parser.parse_args()
        assert args.save_full_image is False

    with patch("sys.argv", mflux_catvton_minimal_argv + ["--save-full-image"]):
        args = mflux_catvton_parser.parse_args()
        assert args.save_full_image is True
