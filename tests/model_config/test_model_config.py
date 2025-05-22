import pytest

from mflux.config.model_config import ModelConfig
from mflux.error.error import InvalidBaseModel, ModelConfigError


def test_bfl_dev():
    model = ModelConfig.from_name("dev")
    assert model.alias == "dev"
    assert model.model_name.startswith("black-forest-labs/")
    assert model.max_sequence_length == 512
    assert model.num_train_steps == 1000
    assert model.supports_guidance is True
    assert model.requires_sigma_shift is True


def test_bfl_dev_full_name():
    model = ModelConfig.from_name("black-forest-labs/FLUX.1-dev")
    assert model.alias == "dev"
    assert model.model_name.startswith("black-forest-labs/")
    assert model.max_sequence_length == 512
    assert model.num_train_steps == 1000
    assert model.supports_guidance is True
    assert model.requires_sigma_shift is True


def test_bfl_schnell():
    model = ModelConfig.from_name("schnell")
    assert model.alias == "schnell"
    assert model.model_name.startswith("black-forest-labs/")
    assert model.max_sequence_length == 256
    assert model.num_train_steps == 1000
    assert model.supports_guidance is False
    assert model.requires_sigma_shift is False


def test_bfl_schnell_full_name():
    model = ModelConfig.from_name("black-forest-labs/FLUX.1-schnell")
    assert model.alias == "schnell"
    assert model.model_name.startswith("black-forest-labs/")
    assert model.max_sequence_length == 256
    assert model.num_train_steps == 1000
    assert model.supports_guidance is False
    assert model.requires_sigma_shift is False


def test_bfl_dev_fill():
    model = ModelConfig.from_name("dev-fill")
    assert model.alias == "dev-fill"
    assert model.model_name == "black-forest-labs/FLUX.1-Fill-dev"
    assert model.max_sequence_length == 512
    assert model.num_train_steps == 1000
    assert model.supports_guidance is True
    assert model.requires_sigma_shift is True


def test_bfl_dev_fill_full_name():
    model = ModelConfig.from_name("black-forest-labs/FLUX.1-Fill-dev")
    assert model.alias == "dev-fill"
    assert model.model_name == "black-forest-labs/FLUX.1-Fill-dev"
    assert model.max_sequence_length == 512
    assert model.num_train_steps == 1000
    assert model.supports_guidance is True
    assert model.requires_sigma_shift is True


def test_bfl_dev_depth():
    model = ModelConfig.from_name("dev-depth")
    assert model.alias == "dev-depth"
    assert model.model_name == "black-forest-labs/FLUX.1-Depth-dev"
    assert model.max_sequence_length == 512
    assert model.num_train_steps == 1000
    assert model.supports_guidance is True
    assert model.requires_sigma_shift is True


def test_bfl_dev_depth_full_name():
    model = ModelConfig.from_name("black-forest-labs/FLUX.1-Depth-dev")
    assert model.alias == "dev-depth"
    assert model.model_name == "black-forest-labs/FLUX.1-Depth-dev"
    assert model.max_sequence_length == 512
    assert model.num_train_steps == 1000
    assert model.supports_guidance is True
    assert model.requires_sigma_shift is True


def test_bfl_dev_redux():
    model = ModelConfig.from_name("dev-redux")
    assert model.alias == "dev-redux"
    assert model.model_name == "black-forest-labs/FLUX.1-Redux-dev"
    assert model.max_sequence_length == 512
    assert model.num_train_steps == 1000
    assert model.supports_guidance is True
    assert model.requires_sigma_shift is True


def test_bfl_dev_redux_full_name():
    model = ModelConfig.from_name("black-forest-labs/FLUX.1-Redux-dev")
    assert model.alias == "dev-redux"
    assert model.model_name == "black-forest-labs/FLUX.1-Redux-dev"
    assert model.max_sequence_length == 512
    assert model.num_train_steps == 1000
    assert model.supports_guidance is True
    assert model.requires_sigma_shift is True


def test_bfl_dev_controlnet_canny():
    model = ModelConfig.from_name("dev-controlnet-canny")
    assert model.alias == "dev-controlnet-canny"
    assert model.model_name == "black-forest-labs/FLUX.1-dev"
    assert model.controlnet_model == "InstantX/FLUX.1-dev-Controlnet-Canny"
    assert model.max_sequence_length == 512
    assert model.num_train_steps == 1000
    assert model.supports_guidance is True
    assert model.requires_sigma_shift is True
    assert model.is_canny() is True


def test_bfl_schnell_controlnet_canny():
    model = ModelConfig.from_name("schnell-controlnet-canny")
    assert model.alias == "schnell-controlnet-canny"
    assert model.model_name == "black-forest-labs/FLUX.1-schnell"
    assert model.controlnet_model == "InstantX/FLUX.1-dev-Controlnet-Canny"
    assert model.max_sequence_length == 256
    assert model.num_train_steps == 1000
    assert model.supports_guidance is False
    assert model.requires_sigma_shift is False
    assert model.is_canny() is True


def test_bfl_dev_controlnet_upscaler():
    model = ModelConfig.from_name("dev-controlnet-upscaler")
    assert model.alias == "dev-controlnet-upscaler"
    assert model.model_name == "black-forest-labs/FLUX.1-dev"
    assert model.controlnet_model == "jasperai/Flux.1-dev-Controlnet-Upscaler"
    assert model.max_sequence_length == 512
    assert model.num_train_steps == 1000
    assert model.supports_guidance is False
    assert model.requires_sigma_shift is False
    assert model.is_canny() is False


def test_community_dev_fill_implicit_base_model():
    model = ModelConfig.from_name("acme-lab/some-dev-fill-model")
    assert model.alias == "dev-fill"
    assert model.max_sequence_length == 512
    assert model.num_train_steps == 1000
    assert model.supports_guidance is True
    assert model.requires_sigma_shift is True


def test_community_dev_fill_explicit_base_model():
    model = ModelConfig.from_name("acme-lab/some-model", base_model="dev-fill")
    assert model.alias == "dev-fill"
    assert model.max_sequence_length == 512
    assert model.num_train_steps == 1000
    assert model.supports_guidance is True
    assert model.requires_sigma_shift is True


def test_implicit_base_model_prefers_dev_fill_over_dev():
    model = ModelConfig.from_name("acme-lab/dev-fill-based-model")
    assert model.alias == "dev-fill"
    assert model.base_model == "black-forest-labs/FLUX.1-Fill-dev"
    assert model.max_sequence_length == 512
    assert model.num_train_steps == 1000
    assert model.requires_sigma_shift is True


def test_community_dev_implicit_base_model():
    model = ModelConfig.from_name("acme-lab/some-awesome-dev-model")
    assert model.alias == "dev"
    assert model.max_sequence_length == 512
    assert model.num_train_steps == 1000
    assert model.supports_guidance is True
    assert model.requires_sigma_shift is True


def test_community_schnell_implicit_base_model():
    model = ModelConfig.from_name("acme-lab/some-quick-schnell-model")
    assert model.alias == "schnell"
    assert model.max_sequence_length == 256
    assert model.num_train_steps == 1000
    assert model.supports_guidance is False
    assert model.requires_sigma_shift is False


def test_community_dev_explicit_base_model():
    model = ModelConfig.from_name("acme-lab/some-awesome-model", base_model="dev")
    assert model.alias == "dev"
    assert model.base_model == "black-forest-labs/FLUX.1-dev"
    assert model.max_sequence_length == 512
    assert model.num_train_steps == 1000
    assert model.supports_guidance is True
    assert model.requires_sigma_shift is True


def test_community_schnell_explicit_base_model():
    model = ModelConfig.from_name("acme-lab/some-awesome-model", base_model="schnell")
    assert model.base_model == "black-forest-labs/FLUX.1-schnell"
    assert model.max_sequence_length == 256
    assert model.num_train_steps == 1000
    assert model.supports_guidance is False
    assert model.requires_sigma_shift is False


def test_model_config_error():
    with pytest.raises(ModelConfigError):
        ModelConfig.from_name("acme-lab/some-model-who-knows-what-its-based-on")


def test_invalid_base_model_error():
    with pytest.raises(InvalidBaseModel):
        ModelConfig.from_name("acme-lab/some-model-who-knows-what-its-based-on", base_model="something_unknown")
