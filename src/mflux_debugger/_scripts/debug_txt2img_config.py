from dataclasses import dataclass


@dataclass(frozen=True)
class DebugTxt2ImgConfig:
    prompt: str = "A hyper-detailed, ultra-fluffy owl sitting in the trees at night, looking directly at the camera with wide, adorable, expressive eyes. Its feathers are soft and voluminous, catching the cool moonlight with subtle silver highlights. The owl's gaze is curious and full of charm, giving it a whimsical, storybook-like personality."
    negative_prompt: str = "ugly, blurry, low quality"
    seed: int = 42
    height: int = 176
    width: int = 320
    num_inference_steps: int = 20
    guidance: float = 4.0


TXT2IMG_DEBUG_CONFIG = DebugTxt2ImgConfig()
