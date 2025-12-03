import pytest

from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.utils.exceptions import InvalidBaseModel, ModelConfigError


class TestConfigResolutionExactMatch:
    @pytest.mark.fast
    def test_exact_alias_match(self):
        config = ConfigResolution.resolve(model_name="schnell")

        assert config.model_name == "black-forest-labs/FLUX.1-schnell"
        assert "schnell" in config.aliases

    @pytest.mark.fast
    def test_exact_alias_match_dev(self):
        config = ConfigResolution.resolve(model_name="dev")

        assert config.model_name == "black-forest-labs/FLUX.1-dev"

    @pytest.mark.fast
    def test_exact_alias_match_fibo(self):
        config = ConfigResolution.resolve(model_name="fibo")

        assert config.model_name == "briaai/FIBO"

    @pytest.mark.fast
    def test_exact_hf_name_match(self):
        config = ConfigResolution.resolve(model_name="black-forest-labs/FLUX.1-schnell")

        assert config.model_name == "black-forest-labs/FLUX.1-schnell"


class TestConfigResolutionExplicitBase:
    @pytest.mark.fast
    def test_explicit_base_model(self):
        config = ConfigResolution.resolve(model_name="my-custom-model", base_model="schnell")

        assert config.model_name == "my-custom-model"
        assert config.base_model == "black-forest-labs/FLUX.1-schnell"
        assert config.max_sequence_length == 256  # schnell's value

    @pytest.mark.fast
    def test_explicit_base_model_dev(self):
        config = ConfigResolution.resolve(model_name="org/my-finetune", base_model="dev")

        assert config.model_name == "org/my-finetune"
        assert config.base_model == "black-forest-labs/FLUX.1-dev"
        assert config.supports_guidance is True  # dev's value

    @pytest.mark.fast
    def test_invalid_base_model_raises(self):
        with pytest.raises(InvalidBaseModel):
            ConfigResolution.resolve(model_name="whatever", base_model="invalid-base")


class TestConfigResolutionInferSubstring:
    @pytest.mark.fast
    def test_infer_from_schnell_substring(self):
        config = ConfigResolution.resolve(model_name="my-schnell-finetune")

        assert config.model_name == "my-schnell-finetune"
        assert config.base_model == "black-forest-labs/FLUX.1-schnell"

    @pytest.mark.fast
    def test_infer_from_dev_substring(self):
        config = ConfigResolution.resolve(model_name="dev-lora-something")

        assert config.model_name == "dev-lora-something"
        assert config.base_model == "black-forest-labs/FLUX.1-dev"

    @pytest.mark.fast
    def test_infer_case_insensitive(self):
        config = ConfigResolution.resolve(model_name="MY-SCHNELL-MODEL")

        assert config.base_model == "black-forest-labs/FLUX.1-schnell"

    @pytest.mark.fast
    def test_longer_alias_preferred(self):
        # "dev-kontext" is longer than "dev", should match dev-kontext if present
        config = ConfigResolution.resolve(model_name="my-dev-kontext-model")

        assert config.base_model == "black-forest-labs/FLUX.1-Kontext-dev"


class TestConfigResolutionError:
    @pytest.mark.fast
    def test_unknown_model_without_base_raises(self):
        with pytest.raises(ModelConfigError) as exc_info:
            ConfigResolution.resolve(model_name="totally-unknown-model")

        assert "Cannot infer" in str(exc_info.value)


class TestConfigResolutionRules:
    @pytest.mark.fast
    def test_exact_match_takes_priority(self):
        # "schnell" is both an exact alias AND would match substring
        config = ConfigResolution.resolve(model_name="schnell")

        # Should return the exact config, not create a new one
        assert config.model_name == "black-forest-labs/FLUX.1-schnell"

    @pytest.mark.fast
    def test_explicit_base_overrides_inference(self):
        # Model name contains "schnell" but explicit base is "dev"
        config = ConfigResolution.resolve(model_name="schnell-style-dev", base_model="dev")

        assert config.base_model == "black-forest-labs/FLUX.1-dev"
