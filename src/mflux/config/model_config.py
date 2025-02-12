from functools import lru_cache
from typing import Literal

from mflux.error.error import InvalidBaseModel, ModelConfigError


class ModelConfig:
    def __init__(
        self,
        alias: str | None,
        model_name: str,
        base_model: str | None,
        num_train_steps: int,
        max_sequence_length: int,
        supports_guidance: bool,
    ):
        self.alias = alias
        self.model_name = model_name
        self.base_model = base_model
        self.num_train_steps = num_train_steps
        self.max_sequence_length = max_sequence_length
        self.supports_guidance = supports_guidance

    @staticmethod
    @lru_cache
    def dev() -> "ModelConfig":
        return ModelConfig(
            alias="dev",
            model_name="black-forest-labs/FLUX.1-dev",
            base_model=None,
            num_train_steps=1000,
            max_sequence_length=512,
            supports_guidance=True,
        )

    @staticmethod
    @lru_cache
    def schnell() -> "ModelConfig":
        return ModelConfig(
            alias="schnell",
            model_name="black-forest-labs/FLUX.1-schnell",
            base_model=None,
            num_train_steps=1000,
            max_sequence_length=256,
            supports_guidance=False,
        )

    @staticmethod
    def from_name(
        model_name: str,
        base_model: Literal["dev", "schnell"] | None = None,
    ) -> "ModelConfig":
        dev = ModelConfig.dev()
        schnell = ModelConfig.schnell()

        # 0. Validate explicit base_model
        allowed_names = [dev.alias, dev.model_name, schnell.alias, schnell.model_name]
        if base_model and base_model not in allowed_names:
            raise InvalidBaseModel(f"Invalid base_model. Choose one of {allowed_names}")

        # 1. If model_name is "dev" or "schnell" then simply return
        if model_name == dev.model_name or model_name == dev.alias:
            return dev
        if model_name == schnell.model_name or model_name == schnell.alias:
            return schnell

        # 1. Determine the appropriate base model
        default_base = None
        if not base_model:
            if "dev" in model_name:
                default_base = dev
            elif "schnell" in model_name:
                default_base = schnell
            else:
                raise ModelConfigError(f"Cannot infer base_model from {model_name}. Specify --base-model.")
        elif base_model == dev.model_name or base_model == dev.alias:
            default_base = dev
        elif base_model == schnell.model_name or base_model == schnell.alias:
            default_base = schnell

        # 2. Construct the config based on the model name and base default
        return ModelConfig(
            alias=default_base.alias,
            model_name=model_name,
            base_model=default_base.model_name,
            num_train_steps=default_base.num_train_steps,
            max_sequence_length=default_base.max_sequence_length,
            supports_guidance=default_base.supports_guidance,
        )

    def is_dev(self) -> bool:
        return self.alias == "dev"
