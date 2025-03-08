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
        requires_sigma_shift: bool,
        priority: int,
    ):
        self.alias = alias
        self.model_name = model_name
        self.base_model = base_model
        self.num_train_steps = num_train_steps
        self.max_sequence_length = max_sequence_length
        self.supports_guidance = supports_guidance
        self.requires_sigma_shift = requires_sigma_shift
        self.priority = priority

    @staticmethod
    @lru_cache
    def dev() -> "ModelConfig":
        return AVAILABLE_MODELS["dev"]

    @staticmethod
    @lru_cache
    def dev_fill() -> "ModelConfig":
        return AVAILABLE_MODELS["dev-fill"]

    @staticmethod
    @lru_cache
    def schnell() -> "ModelConfig":
        return AVAILABLE_MODELS["schnell"]

    @staticmethod
    def from_name(
        model_name: str,
        base_model: Literal["dev", "schnell", "dev-fill"] | None = None,
    ) -> "ModelConfig":
        # 0. Get all base models (where base_model is None) sorted by priority
        base_models = sorted(
            [model for model in AVAILABLE_MODELS.values() if model.base_model is None], key=lambda x: x.priority
        )

        # 1. Check if model_name matches any base model's alias or full name
        for base in base_models:
            if model_name in (base.alias, base.model_name):
                return base

        # 2. Validate explicit base_model
        allowed_names = []
        for base in base_models:
            allowed_names.extend([base.alias, base.model_name])
        if base_model and base_model not in allowed_names:
            raise InvalidBaseModel(f"Invalid base_model. Choose one of {allowed_names}")

        # 3. Determine the base model (explicit or inferred)
        if base_model:
            # Find by explicit base_model name
            default_base = next((b for b in base_models if base_model in (b.alias, b.model_name)), None)
        else:
            # Infer from model_name substring (priority order via sorted base_models)
            default_base = next((b for b in base_models if b.alias and b.alias in model_name), None)
            if not default_base:
                raise ModelConfigError(f"Cannot infer base_model from {model_name}")

        # 4. Construct the config
        return ModelConfig(
            alias=default_base.alias,
            model_name=model_name,
            base_model=default_base.model_name,
            num_train_steps=default_base.num_train_steps,
            max_sequence_length=default_base.max_sequence_length,
            supports_guidance=default_base.supports_guidance,
            requires_sigma_shift=default_base.requires_sigma_shift,
            priority=default_base.priority,
        )


AVAILABLE_MODELS = {
    "schnell": ModelConfig(
        alias="schnell",
        model_name="black-forest-labs/FLUX.1-schnell",
        base_model=None,
        num_train_steps=1000,
        max_sequence_length=256,
        supports_guidance=False,
        requires_sigma_shift=False,
        priority=2,
    ),
    "dev": ModelConfig(
        alias="dev",
        model_name="black-forest-labs/FLUX.1-dev",
        base_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=True,
        priority=1,
    ),
    "dev-fill": ModelConfig(
        alias="dev-fill",
        model_name="black-forest-labs/FLUX.1-Fill-dev",
        base_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=True,
        priority=0,
    ),
}
