from pathlib import Path

import PIL.Image
import pytest

from mflux.cli.defaults import defaults as ui_defaults
from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig


@pytest.mark.fast
def test_partial_width_none_resolves_from_reference_image(tmp_path: Path):
    image_path = tmp_path / "reference.png"
    PIL.Image.new("RGB", (1200, 800)).save(image_path)

    config = Config(
        model_config=ModelConfig.flux2_klein_4b(),
        width=None,
        height=512,
        image_path=image_path,
    )

    assert config.width == 1200
    assert config.height == 512


@pytest.mark.fast
def test_partial_height_none_resolves_from_defaults_without_reference_image():
    config = Config(
        model_config=ModelConfig.flux2_klein_4b(),
        width=640,
        height=None,
        image_path=None,
    )

    assert config.width == 640
    assert config.height == ui_defaults.HEIGHT


@pytest.mark.fast
def test_both_dimensions_none_resolve_from_reference_image(tmp_path: Path):
    image_path = tmp_path / "reference.png"
    PIL.Image.new("RGB", (1200, 800)).save(image_path)

    config = Config(
        model_config=ModelConfig.flux2_klein_4b(),
        width=None,
        height=None,
        image_path=image_path,
    )

    assert config.width == 1200
    assert config.height == 800


@pytest.mark.fast
def test_both_dimensions_none_resolve_to_defaults_without_reference_image():
    config = Config(
        model_config=ModelConfig.flux2_klein_4b(),
        width=None,
        height=None,
        image_path=None,
    )

    assert config.width == ui_defaults.WIDTH
    assert config.height == ui_defaults.HEIGHT
