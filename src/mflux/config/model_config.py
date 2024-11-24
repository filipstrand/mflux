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

    @property
    def alias(self):
        # maintain compatibility with < 0.4.0 behavior
        # where alias is the name of an official model
        if self.model_name.startswith("black-forest-labs/FLUX.1-"):
            return self.model_name[len("black-forest-labs/FLUX.1-") :].lower()
        return None


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
    def from_name(
        alias: str,
        base_model: Literal["dev", "schnell"] | None = None,
    ) -> ModelConfig:
        if alias in DefaultModelConfigs:
            return DefaultModelConfigs[alias]

        if all(["dev" not in alias, "schnell" not in alias, base_model is None]):
            raise ModelConfigError(
                "Cannot infer base model and max_sequence_length "
                f"from model reference: {alias!r}. "
                "Please specify --base-model [dev | schnell]"
            )

        if base_model is not None and base_model not in ["dev", "schnell"]:
            raise InvalidBaseModel("As of this version, mflux only recognizes base models dev or schnell")

        if base_model is None:
            # infer base model on apparent model namming
            if "dev" in alias:
                base_model = "dev"
            elif "schnell" in alias:
                base_model = "schnell"

        if base_model == "dev":
            supports_guidance = True
            max_sequence_length = KNOWN_SEQUENCE_LENGTH_BY_BASE_MODEL["dev"]
        elif base_model == "schnell":
            supports_guidance = False
            max_sequence_length = KNOWN_SEQUENCE_LENGTH_BY_BASE_MODEL["schnell"]

        return ModelConfig(
            alias,  # actually this arg is model_name
            DEFAULT_TRAIN_STEPS,
            max_sequence_length,
            supports_guidance,
            base_model,
        )


# maintain old `ModelConfig.from_alias` function name for backwards compatibility in user code and docs
ModelConfig.from_alias = ModelLookup.from_name
# keep these class members to be backwards compatible with < 0.5.0 ModelConfig Enum implementation
ModelConfig.FLUX1_DEV = DefaultModelConfigs["dev"]
ModelConfig.FLUX1_SCHNELL = DefaultModelConfigs["schnell"]
