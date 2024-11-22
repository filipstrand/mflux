from dataclasses import dataclass
from typing import Literal


class ModelConfigError(ValueError):
    pass


DEFAULT_TRAIN_STEPS = 1000


@dataclass
class ModelAttrs:
    model_name: str
    num_train_steps: int
    max_sequence_length: int
    supports_guidance: bool

    @property
    def alias(self):
        # maintain compatibility with < 0.4.0 behavior
        # where alias is the name of an official model
        if self.model_name.startswith("black-forest-labs/FLUX.1-"):
            return self.model_name[len("black-forest-labs/FLUX.1-"):].lower()


DefaultModelConfigs = {
    "dev": ModelAttrs(
        model_name="black-forest-labs/FLUX.1-dev",
        num_train_steps=DEFAULT_TRAIN_STEPS,
        max_sequence_length=512,
        supports_guidance=True,
    ),
    "schnell": ModelAttrs(
        model_name="black-forest-labs/FLUX.1-schnell",
        num_train_steps=DEFAULT_TRAIN_STEPS,
        max_sequence_length=256,
        supports_guidance=False,
    ),
}


class ModelConfig:
    # keep these class members to be backwards compatible with < 0.5.0 ModelConfig Enum implementation
    FLUX1_DEV = DefaultModelConfigs["dev"]
    FLUX1_SCHNELL = DefaultModelConfigs["schnell"]

    @staticmethod
    def from_alias(alias: str, base_model: Literal["dev", "schnell"] | None = None) -> ModelAttrs:
        if alias in DefaultModelConfigs:
            return DefaultModelConfigs[alias]

        if all(["dev" not in alias, "schnell" not in alias, base_model is None]):
            raise ModelConfigError(
                "Cannot infer base model and max_sequence_length "
                f"from model reference: {alias!r}. "
                "Please specify --base-model [dev | schnell]"
            )

        if base_model == "dev" or "dev" in alias:
            supports_guidance = True
            max_sequence_length = 512
        elif base_model == "schnell" or "schnell" in alias:
            supports_guidance = False
            max_sequence_length = 256

        return ModelAttrs(
            alias,  # actually this arg is model_name
            DEFAULT_TRAIN_STEPS,
            max_sequence_length,
            supports_guidance,
        )
