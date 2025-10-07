import logging
from pathlib import Path

import mlx.core as mx

from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.schedulers import SCHEDULER_REGISTRY, try_import_external_scheduler
from mflux.schedulers.linear_scheduler import LinearScheduler

logger = logging.getLogger(__name__)


class RuntimeConfig:
    def __init__(
        self,
        config: Config,
        model_config: ModelConfig,
    ):
        self.config = config
        self.model_config = model_config
        self._scheduler = None

    @property
    def height(self) -> int:
        return self.config.height

    @property
    def width(self) -> int:
        return self.config.width

    @width.setter
    def width(self, value):
        self.config.width = value

    @property
    def guidance(self) -> float:
        return self.config.guidance

    @property
    def num_inference_steps(self) -> int:
        return self.config.num_inference_steps

    @property
    def precision(self) -> mx.Dtype:
        return self.config.precision

    @property
    def num_train_steps(self) -> int:
        return self.model_config.num_train_steps

    @property
    def image_path(self) -> Path | None:
        return self.config.image_path

    @property
    def image_strength(self) -> float | None:
        return self.config.image_strength

    @property
    def depth_image_path(self) -> Path | None:
        return self.config.depth_image_path

    @property
    def redux_image_paths(self) -> list[Path] | None:
        return self.config.redux_image_paths

    @property
    def redux_image_strengths(self) -> list[float] | None:
        return self.config.redux_image_strengths

    @property
    def masked_image_path(self) -> Path | None:
        return self.config.masked_image_path

    @property
    def init_time_step(self) -> int:
        is_img2img = (
            self.config.image_path is not None and
            self.image_strength is not None and
            self.image_strength > 0.0
        )  # fmt: off

        if is_img2img:
            # 1. Clamp strength to [0, 1]
            strength = max(0.0, min(1.0, self.config.image_strength))  # type: ignore

            # 2. Return start time in [1, floor(num_steps * strength)]
            return max(1, int(self.num_inference_steps * strength))  # type: ignore
        else:
            return 0

    @property
    def controlnet_strength(self) -> float | None:
        if self.config.controlnet_strength is not None:
            return self.config.controlnet_strength

        return None

    @property
    def scheduler(self):
        if self._scheduler is not None:
            return self._scheduler

        if self.config.scheduler_str == "linear":
            self._scheduler = LinearScheduler(self)
        elif (registered_scheduler := SCHEDULER_REGISTRY.get(self.config.scheduler_str, None)) is not None:
            self._scheduler = registered_scheduler
        elif "." in self.config.scheduler_str:
            # this raises ValueError if scheduler is not importable
            scheduler_cls = try_import_external_scheduler(self.config.scheduler_str)
            self._scheduler = scheduler_cls(self)
        else:
            raise NotImplementedError(f"The scheduler {self.config.scheduler_str!r} is not implemented by mflux.")

        return self._scheduler
