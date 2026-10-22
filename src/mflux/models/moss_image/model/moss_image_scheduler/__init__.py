from mflux.models.common.schedulers import register_contrib
from mflux.models.moss_image.model.moss_image_scheduler.moss_image_scheduler import MOSSImageScheduler

register_contrib(MOSSImageScheduler, "moss_image")

__all__ = ["MOSSImageScheduler"]
