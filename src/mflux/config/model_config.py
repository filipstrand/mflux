import warnings
from dataclasses import dataclass
from typing import Literal

DEFAULT_TRAIN_STEPS = 1000

KNOWN_SEQUENCE_LENGTH_BY_BASE_MODEL = {"dev": 512, "schnell": 256}


class ModelConfigError(ValueError):
    """User error in model config."""


class InvalidBaseModel(ModelConfigError):
    """Invalid base model, cannot infer model properties."""


@dataclass
class ModelConfig:
    model_name: str
    num_train_steps: int
    max_sequence_length: int
    supports_guidance: bool
    base_model: str | None


DefaultModelConfigs = {
    "dev": ModelConfig(
        model_name="black-forest-labs/FLUX.1-dev",
        num_train_steps=DEFAULT_TRAIN_STEPS,
        max_sequence_length=KNOWN_SEQUENCE_LENGTH_BY_BASE_MODEL["dev"],
        supports_guidance=True,
        base_model=None,
    ),
    "schnell": ModelConfig(
        model_name="black-forest-labs/FLUX.1-schnell",
        num_train_steps=DEFAULT_TRAIN_STEPS,
        max_sequence_length=KNOWN_SEQUENCE_LENGTH_BY_BASE_MODEL["schnell"],
        supports_guidance=False,
        base_model=None,
    ),
}


class ModelLookup:
    @staticmethod
    def from_alias(alias: str) -> ModelConfig:
        warnings.warn(
            "from_alias is deprecated and will be removed in a future release. " "Please use from_name instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return ModelLookup.from_name(model_name=alias, base_model=None)

    @staticmethod
    def from_name(
        model_name: str,
        base_model: Literal["dev", "schnell"] | None = None,
    ) -> ModelConfig:
        if model_name in DefaultModelConfigs:
            return DefaultModelConfigs[model_name]

        if all(["dev" not in model_name, "schnell" not in model_name, base_model is None]):
            raise ModelConfigError(
                "Cannot infer base model and max_sequence_length "
                f"from model reference: {model_name!r}. "
                "Please specify --base-model [dev | schnell]"
            )

        if base_model is not None and base_model not in ["dev", "schnell"]:
            raise InvalidBaseModel("As of this version, mflux only recognizes base models dev or schnell")

        if base_model is None:
            # infer base model on apparent model naming
            if "dev" in model_name:
                base_model = "dev"
            elif "schnell" in model_name:
                base_model = "schnell"

        if base_model == "dev":
            supports_guidance = True
            max_sequence_length = KNOWN_SEQUENCE_LENGTH_BY_BASE_MODEL["dev"]
        elif base_model == "schnell":
            supports_guidance = False
            max_sequence_length = KNOWN_SEQUENCE_LENGTH_BY_BASE_MODEL["schnell"]

        return ModelConfig(
            model_name=model_name,
            num_train_steps=DEFAULT_TRAIN_STEPS,
            max_sequence_length=max_sequence_length,
            supports_guidance=supports_guidance,
            base_model=base_model,
        )


# keep these class members to be backwards compatible with < 0.5.0 ModelConfig Enum implementation
ModelConfig.FLUX1_DEV = DefaultModelConfigs["dev"]
ModelConfig.FLUX1_SCHNELL = DefaultModelConfigs["schnell"]
