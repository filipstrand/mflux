from functools import lru_cache
from typing import Literal

from mflux.error.error import InvalidBaseModel, ModelConfigError


class ModelConfig:
    def __init__(
        self,
        aliases: list[str],
        model_name: str,
        base_model: str | None,
        controlnet_model: str | None,
        custom_transformer_model: str | None,
        num_train_steps: int,
        max_sequence_length: int,
        supports_guidance: bool,
        requires_sigma_shift: bool,
        priority: int,
    ):
        self.aliases = aliases
        self.model_name = model_name
        self.base_model = base_model
        self.controlnet_model = controlnet_model
        self.custom_transformer_model = custom_transformer_model
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
    def schnell() -> "ModelConfig":
        return AVAILABLE_MODELS["schnell"]

    @staticmethod
    @lru_cache
    def dev_kontext() -> "ModelConfig":
        return AVAILABLE_MODELS["dev-kontext"]

    @staticmethod
    @lru_cache
    def dev_fill() -> "ModelConfig":
        return AVAILABLE_MODELS["dev-fill"]

    @staticmethod
    @lru_cache
    def dev_redux() -> "ModelConfig":
        return AVAILABLE_MODELS["dev-redux"]

    @staticmethod
    @lru_cache
    def dev_depth() -> "ModelConfig":
        return AVAILABLE_MODELS["dev-depth"]

    @staticmethod
    @lru_cache
    def dev_controlnet_canny() -> "ModelConfig":
        return AVAILABLE_MODELS["dev-controlnet-canny"]

    @staticmethod
    @lru_cache
    def schnell_controlnet_canny() -> "ModelConfig":
        return AVAILABLE_MODELS["schnell-controlnet-canny"]

    @staticmethod
    @lru_cache
    def dev_controlnet_upscaler() -> "ModelConfig":
        return AVAILABLE_MODELS["dev-controlnet-upscaler"]

    @staticmethod
    @lru_cache
    def dev_fill_catvton() -> "ModelConfig":
        return AVAILABLE_MODELS["dev-fill-catvton"]

    @staticmethod
    @lru_cache
    def krea_dev() -> "ModelConfig":
        return AVAILABLE_MODELS["krea-dev"]

    def x_embedder_input_dim(self) -> int:
        if "Fill" in self.model_name:
            return 384
        if "Depth" in self.model_name:
            return 128
        else:
            return 64

    def is_canny(self) -> bool:
        return self.controlnet_model is not None and "Canny" in self.controlnet_model

    @staticmethod
    def from_name(
        model_name: str,
        base_model: Literal["dev", "schnell", "krea-dev"] | None = None,
    ) -> "ModelConfig":
        # 0. Get all base models (where base_model is None) sorted by priority
        base_models = sorted(
            [model for model in AVAILABLE_MODELS.values() if model.base_model is None], key=lambda x: x.priority
        )

        # 1. Check if model_name matches any base model's aliases or full name
        for base in base_models:
            if model_name == base.model_name or model_name in base.aliases:
                return base

        # 2. Validate explicit base_model
        allowed_names = []
        for base in base_models:
            allowed_names.extend(base.aliases + [base.model_name])
        if base_model and base_model not in allowed_names:
            raise InvalidBaseModel(f"Invalid base_model. Choose one of {allowed_names}")

        # 3. Determine the base model (explicit or inferred)
        if base_model:
            # Find by explicit base_model name (check all aliases)
            default_base = next((b for b in base_models if base_model == b.model_name or base_model in b.aliases), None)
        else:
            # Infer from model_name substring - prefer longer matches (more specific)
            matching_bases = [(b, alias) for b in base_models for alias in b.aliases if alias and alias in model_name]

            if matching_bases:
                # Sort by alias length descending, then by priority ascending
                default_base = sorted(matching_bases, key=lambda x: (-len(x[1]), x[0].priority))[0][0]
            else:
                default_base = None
            if not default_base:
                raise ModelConfigError(f"Cannot infer base_model from {model_name}")

        # 4. Construct the config
        return ModelConfig(
            aliases=default_base.aliases,
            model_name=model_name,
            base_model=default_base.model_name,
            controlnet_model=default_base.controlnet_model,
            custom_transformer_model=default_base.custom_transformer_model,
            num_train_steps=default_base.num_train_steps,
            max_sequence_length=default_base.max_sequence_length,
            supports_guidance=default_base.supports_guidance,
            requires_sigma_shift=default_base.requires_sigma_shift,
            priority=default_base.priority,
        )


AVAILABLE_MODELS = {
    "dev": ModelConfig(
        aliases=["dev"],
        model_name="black-forest-labs/FLUX.1-dev",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=True,
        priority=0,
    ),
    "schnell": ModelConfig(
        aliases=["schnell"],
        model_name="black-forest-labs/FLUX.1-schnell",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=256,
        supports_guidance=False,
        requires_sigma_shift=False,
        priority=1,
    ),
    "dev-kontext": ModelConfig(
        aliases=["dev-kontext"],
        model_name="black-forest-labs/FLUX.1-Kontext-dev",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=True,
        priority=2,
    ),
    "dev-fill": ModelConfig(
        aliases=["dev-fill"],
        model_name="black-forest-labs/FLUX.1-Fill-dev",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=True,
        priority=3,
    ),
    "dev-redux": ModelConfig(
        aliases=["dev-redux"],
        model_name="black-forest-labs/FLUX.1-Redux-dev",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=True,
        priority=4,
    ),
    "dev-depth": ModelConfig(
        aliases=["dev-depth"],
        model_name="black-forest-labs/FLUX.1-Depth-dev",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=True,
        priority=5,
    ),
    "dev-controlnet-canny": ModelConfig(
        aliases=["dev-controlnet-canny"],
        model_name="black-forest-labs/FLUX.1-dev",
        base_model=None,
        controlnet_model="InstantX/FLUX.1-dev-Controlnet-Canny",
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=True,
        priority=6,
    ),
    "schnell-controlnet-canny": ModelConfig(
        aliases=["schnell-controlnet-canny"],
        model_name="black-forest-labs/FLUX.1-schnell",
        base_model=None,
        controlnet_model="InstantX/FLUX.1-dev-Controlnet-Canny",
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=256,
        supports_guidance=False,
        requires_sigma_shift=False,
        priority=7,
    ),
    "dev-controlnet-upscaler": ModelConfig(
        aliases=["dev-controlnet-upscaler"],
        model_name="black-forest-labs/FLUX.1-dev",
        base_model=None,
        controlnet_model="jasperai/Flux.1-dev-Controlnet-Upscaler",
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=False,
        requires_sigma_shift=False,
        priority=8,
    ),
    "dev-fill-catvton": ModelConfig(
        aliases=["dev-fill-catvton"],
        model_name="black-forest-labs/FLUX.1-Fill-dev",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model="xiaozaa/catvton-flux-beta",
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=False,  # Not sure why, but produced better results this way...
        priority=9,
    ),
    "krea-dev": ModelConfig(
        aliases=["krea-dev", "dev-krea"],
        model_name="black-forest-labs/FLUX.1-Krea-dev",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=True,
        priority=10,
    ),
}
