import pytest

from mflux.config.model_config import InvalidBaseModel, ModelConfigError, ModelLookup


def test_bfl_dev():
    model_attrs = ModelLookup.from_name("dev")
    assert model_attrs.model_name.startswith("black-forest-labs/")
    assert model_attrs.max_sequence_length == 512
    assert model_attrs.supports_guidance is True


def test_bfl_dev_from_alias():
    model_attrs = ModelLookup.from_alias("dev")
    assert model_attrs.model_name.startswith("black-forest-labs/")
    assert model_attrs.max_sequence_length == 512
    assert model_attrs.supports_guidance is True


def test_bfl_schnell():
    model_attrs = ModelLookup.from_name("schnell")
    assert model_attrs.model_name.startswith("black-forest-labs/")
    assert model_attrs.max_sequence_length == 256
    assert model_attrs.supports_guidance is False


def test_bfl_schnell_from_alias():
    model_attrs = ModelLookup.from_alias("schnell")
    assert model_attrs.model_name.startswith("black-forest-labs/")
    assert model_attrs.max_sequence_length == 256
    assert model_attrs.supports_guidance is False


def test_community_dev_implicit_base_model():
    model_attrs = ModelLookup.from_name("acme-lab/some-awesome-dev-model")
    assert model_attrs.max_sequence_length == 512
    assert model_attrs.supports_guidance is True


def test_community_schnell_implicit_base_model():
    model_attrs = ModelLookup.from_name("acme-lab/some-quick-schnell-model")
    assert model_attrs.max_sequence_length == 256
    assert model_attrs.supports_guidance is False


def test_community_dev_explicit_base_model():
    model_attrs = ModelLookup.from_name("acme-lab/some-awesome-model", base_model="dev")
    assert model_attrs.max_sequence_length == 512
    assert model_attrs.supports_guidance is True


def test_community_schnell_explicit_base_model():
    model_attrs = ModelLookup.from_name("acme-lab/some-awesome-model", base_model="schnell")
    assert model_attrs.max_sequence_length == 256
    assert model_attrs.supports_guidance is False


def test_model_config_error():
    assert pytest.raises(ModelConfigError, ModelLookup.from_name, "acme-lab/some-model-who-knows-what-its-based-on")


def test_invalid_base_model_error():
    assert pytest.raises(
        InvalidBaseModel,
        ModelLookup.from_name,
        "acme-lab/some-model-who-knows-what-its-based-on",
        base_model="something_unknown",
    )
