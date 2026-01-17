from functools import lru_cache
from typing import Literal

import mlx.core as mx

from mflux.models.common.resolution.config_resolution import ConfigResolution


class ModelConfig:
    precision: mx.Dtype = mx.bfloat16

    def __init__(
        self,
        priority: int,
        aliases: list[str],
        model_name: str,
        base_model: str | None,
        controlnet_model: str | None,
        custom_transformer_model: str | None,
        num_train_steps: int | None,
        max_sequence_length: int | None,
        supports_guidance: bool | None,
        requires_sigma_shift: bool | None,
        transformer_overrides: dict | None = None,
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
        self.transformer_overrides = transformer_overrides or {}

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
    def flux2_klein_4b() -> "ModelConfig":
        return AVAILABLE_MODELS["flux2-klein-4b"]

    @staticmethod
    @lru_cache
    def flux2_klein_9b() -> "ModelConfig":
        return AVAILABLE_MODELS["flux2-klein-9b"]

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

    @staticmethod
    @lru_cache
    def seedvr2_3b() -> "ModelConfig":
        return AVAILABLE_MODELS["seedvr2-3b"]

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
        priority=0,
        aliases=["dev"],
        model_name="black-forest-labs/FLUX.1-dev",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=True,
    ),
    "schnell": ModelConfig(
        priority=1,
        aliases=["schnell"],
        model_name="black-forest-labs/FLUX.1-schnell",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=256,
        supports_guidance=False,
        requires_sigma_shift=False,
    ),
    "dev-kontext": ModelConfig(
        priority=2,
        aliases=["dev-kontext"],
        model_name="black-forest-labs/FLUX.1-Kontext-dev",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=True,
    ),
    "dev-fill": ModelConfig(
        priority=3,
        aliases=["dev-fill"],
        model_name="black-forest-labs/FLUX.1-Fill-dev",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=True,
    ),
    "dev-redux": ModelConfig(
        priority=4,
        aliases=["dev-redux"],
        model_name="black-forest-labs/FLUX.1-Redux-dev",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=True,
    ),
    "dev-depth": ModelConfig(
        priority=5,
        aliases=["dev-depth"],
        model_name="black-forest-labs/FLUX.1-Depth-dev",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=True,
    ),
    "dev-controlnet-canny": ModelConfig(
        priority=6,
        aliases=["dev-controlnet-canny"],
        model_name="black-forest-labs/FLUX.1-dev",
        base_model=None,
        controlnet_model="InstantX/FLUX.1-dev-Controlnet-Canny",
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=True,
    ),
    "schnell-controlnet-canny": ModelConfig(
        priority=7,
        aliases=["schnell-controlnet-canny"],
        model_name="black-forest-labs/FLUX.1-schnell",
        base_model=None,
        controlnet_model="InstantX/FLUX.1-dev-Controlnet-Canny",
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=256,
        supports_guidance=False,
        requires_sigma_shift=False,
    ),
    "dev-controlnet-upscaler": ModelConfig(
        priority=8,
        aliases=["dev-controlnet-upscaler"],
        model_name="black-forest-labs/FLUX.1-dev",
        base_model=None,
        controlnet_model="jasperai/Flux.1-dev-Controlnet-Upscaler",
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=False,
        requires_sigma_shift=False,
    ),
    "dev-fill-catvton": ModelConfig(
        priority=9,
        aliases=["dev-fill-catvton"],
        model_name="black-forest-labs/FLUX.1-Fill-dev",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model="xiaozaa/catvton-flux-beta",
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=False,
    ),
    "krea-dev": ModelConfig(
        priority=10,
        aliases=["krea-dev", "dev-krea"],
        model_name="black-forest-labs/FLUX.1-Krea-dev",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=True,
    ),
    "flux2-klein-4b": ModelConfig(
        priority=11,
        aliases=["flux2-klein-4b", "flux2-klein-4B", "flux2-klein", "klein-4b", "klein-4B"],
        model_name="black-forest-labs/FLUX.2-klein-4B",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=True,
        transformer_overrides={
            "num_layers": 5,
            "num_single_layers": 20,
            "num_attention_heads": 24,
            "joint_attention_dim": 7680,
        },
    ),
    "flux2-klein-9b": ModelConfig(
        priority=12,
        aliases=["flux2-klein-9b", "flux2-klein-9B", "klein-9b", "klein-9B"],
        model_name="black-forest-labs/FLUX.2-klein-9B",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=True,
        transformer_overrides={
            "num_layers": 8,
            "num_single_layers": 24,
            "num_attention_heads": 32,
            "joint_attention_dim": 12288,
        },
    ),
    "qwen-image": ModelConfig(
        priority=13,
        aliases=["qwen-image", "qwen"],
        model_name="Qwen/Qwen-Image",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=None,
        max_sequence_length=None,
        supports_guidance=None,
        requires_sigma_shift=None,
    ),
    "qwen-image-edit": ModelConfig(
        priority=14,
        aliases=["qwen-image-edit", "qwen-edit", "qwen-edit-plus", "qwen-edit-2509"],
        model_name="Qwen/Qwen-Image-Edit-2509",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=None,
        max_sequence_length=None,
        supports_guidance=None,
        requires_sigma_shift=None,
    ),
    "fibo": ModelConfig(
        priority=15,
        aliases=["fibo"],
        model_name="briaai/FIBO",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=False,
    ),
    "z-image-turbo": ModelConfig(
        priority=16,
        aliases=["z-image-turbo", "z-image", "zimage-turbo", "zimage"],
        model_name="Tongyi-MAI/Z-Image-Turbo",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=False,  # Turbo model uses guidance_scale=0
        requires_sigma_shift=True,
    ),
    "seedvr2-3b": ModelConfig(
        priority=17,
        aliases=["seedvr2-3b", "seedvr2"],
        model_name="numz/SeedVR2_comfyUI",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=None,
        max_sequence_length=None,
        supports_guidance=True,
        requires_sigma_shift=None,
    ),
}
