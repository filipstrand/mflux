# Import NoiseSchedulerType enum
from mflux.config.constants import NoiseSchedulerType

CONTROLNET_STRENGTH = 0.4
GUIDANCE_SCALE = 3.5
HEIGHT, WIDTH = 1024, 1024
IMAGE_STRENGTH = 0.4
MODEL_CHOICES = ["dev", "schnell"]
MODEL_INFERENCE_STEPS = {
    "dev": 14,
    "schnell": 4,
}
QUANTIZE_CHOICES = [3, 4, 6, 8]
DEFAULT_SCHEDULER = NoiseSchedulerType.LINEAR
SCHEDULER_CHOICES = NoiseSchedulerType.choices()
