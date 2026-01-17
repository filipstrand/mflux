import logging
from pathlib import Path

import mlx.core as mx
from tqdm import tqdm

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.schedulers import SCHEDULER_REGISTRY, try_import_external_scheduler
from mflux.models.common.schedulers.linear_scheduler import LinearScheduler

logger = logging.getLogger(__name__)


class Config:
    def __init__(
        self,
        model_config: ModelConfig,
        num_inference_steps: int = 4,
        height: int = 1024,
        width: int = 1024,
        guidance: float = 4.0,
        image_path: Path | str | None = None,
        image_strength: float | None = None,
        depth_image_path: Path | str | None = None,
        redux_image_paths: list[Path | str] | None = None,
        redux_image_strengths: list[float] | None = None,
        masked_image_path: Path | str | None = None,
        controlnet_strength: float | None = None,
        scheduler: str = "linear",
    ):
        # Ensure dimensions are multiples of 16
        if width % 16 != 0 or height % 16 != 0:
            logger.warning("Width and height should be multiples of 16. Rounding down.")

        self.model_config = model_config
        self._num_inference_steps = num_inference_steps
        self._height = 16 * (height // 16)
        self._width = 16 * (width // 16)
        self._guidance = guidance
        self._image_path = Path(image_path) if isinstance(image_path, str) else image_path
        self._image_strength = image_strength
        self._depth_image_path = Path(depth_image_path) if isinstance(depth_image_path, str) else depth_image_path
        self._redux_image_paths = (
            [Path(p) if isinstance(p, str) else p for p in redux_image_paths] if redux_image_paths else None
        )
        self._redux_image_strengths = redux_image_strengths
        self._masked_image_path = Path(masked_image_path) if isinstance(masked_image_path, str) else masked_image_path
        self._controlnet_strength = controlnet_strength
        self._scheduler_str = scheduler
        self._scheduler = None
        self._time_steps = None

    @property
    def height(self) -> int:
        return self._height

    @property
    def width(self) -> int:
        return self._width

    @width.setter
    def width(self, value):
        self._width = value

    @property
    def image_seq_len(self) -> int:
        return (self._height // 16) * (self._width // 16)

    @property
    def guidance(self) -> float:
        return self._guidance

    @property
    def num_inference_steps(self) -> int:
        return self._num_inference_steps

    @property
    def precision(self) -> mx.Dtype:
        return ModelConfig.precision

    @property
    def num_train_steps(self) -> int:
        return self.model_config.num_train_steps

    @property
    def image_path(self) -> Path | None:
        return self._image_path

    @property
    def image_strength(self) -> float | None:
        return self._image_strength

    @property
    def depth_image_path(self) -> Path | None:
        return self._depth_image_path

    @property
    def redux_image_paths(self) -> list[Path] | None:
        return self._redux_image_paths

    @property
    def redux_image_strengths(self) -> list[float] | None:
        return self._redux_image_strengths

    @property
    def masked_image_path(self) -> Path | None:
        return self._masked_image_path

    @property
    def init_time_step(self) -> int:
        is_img2img = (
            self._image_path is not None and
            self._image_strength is not None and
            self._image_strength > 0.0
        )  # fmt: off

        if is_img2img:
            # 1. Clamp strength to [0, 1]
            strength = max(0.0, min(1.0, self._image_strength))  # type: ignore

            # 2. Return start time in [1, floor(num_steps * strength)]
            return max(1, int(self._num_inference_steps * strength))  # type: ignore
        else:
            return 0

    @property
    def time_steps(self) -> tqdm:
        if self._time_steps is None:
            self._time_steps = tqdm(range(self.init_time_step, self.num_inference_steps))
        return self._time_steps

    @property
    def controlnet_strength(self) -> float | None:
        return self._controlnet_strength

    @property
    def scheduler(self):
        if self._scheduler is not None:
            return self._scheduler

        if self._scheduler_str == "linear":
            self._scheduler = LinearScheduler(self)
        elif (registered_scheduler := SCHEDULER_REGISTRY.get(self._scheduler_str, None)) is not None:
            self._scheduler = registered_scheduler(self)
        elif "." in self._scheduler_str:
            # this raises ValueError if scheduler is not importable
            scheduler_cls = try_import_external_scheduler(self._scheduler_str)
            self._scheduler = scheduler_cls(self)
        else:
            raise NotImplementedError(f"The scheduler {self._scheduler_str!r} is not implemented by mflux.")

        if hasattr(self._scheduler, "set_image_seq_len") and self.model_config.requires_sigma_shift:
            self._scheduler.set_image_seq_len(self.image_seq_len)

        return self._scheduler
