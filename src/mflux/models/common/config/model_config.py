from functools import lru_cache
from typing import Literal

import mlx.core as mx

from mflux.models.common.resolution.config_resolution import ConfigResolution


class ModelConfig:
    precision: mx.Dtype = mx.bfloat16

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

    @staticmethod
    @lru_cache
    def qwen_image() -> "ModelConfig":
        return AVAILABLE_MODELS["qwen-image"]

    @staticmethod
    @lru_cache
    def qwen_image_edit() -> "ModelConfig":
        return AVAILABLE_MODELS["qwen-image-edit"]

    @staticmethod
    @lru_cache
    def fibo() -> "ModelConfig":
        return AVAILABLE_MODELS["fibo"]

    @staticmethod
    @lru_cache
    def z_image_turbo() -> "ModelConfig":
        return AVAILABLE_MODELS["z-image-turbo"]

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
        return ConfigResolution.resolve(model_name=model_name, base_model=base_model)


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
    "qwen-image": ModelConfig(
        aliases=["qwen-image", "qwen"],
        model_name="Qwen/Qwen-Image",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=None,
        max_sequence_length=None,
        supports_guidance=None,
        requires_sigma_shift=None,
        priority=11,
    ),
    "qwen-image-edit": ModelConfig(
        aliases=["qwen-image-edit", "qwen-edit", "qwen-edit-plus", "qwen-edit-2509"],
        model_name="Qwen/Qwen-Image-Edit-2509",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=None,
        max_sequence_length=None,
        supports_guidance=None,
        requires_sigma_shift=None,
        priority=12,
    ),
    "fibo": ModelConfig(
        aliases=["fibo"],
        model_name="briaai/FIBO",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=False,
        priority=13,
    ),
    "z-image-turbo": ModelConfig(
        aliases=["z-image-turbo", "z-image", "zimage-turbo", "zimage"],
        model_name="Tongyi-MAI/Z-Image-Turbo",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=False,  # Turbo model uses guidance_scale=0
        requires_sigma_shift=True,
        priority=14,
    ),
}
