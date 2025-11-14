from dataclasses import dataclass


@dataclass(frozen=True)
class DebugTxt2ImgConfig:
    prompt: str = "A hyper-detailed, ultra-fluffy owl sitting in the trees at night, looking directly at the camera with wide, adorable, expressive eyes. Its feathers are soft and voluminous, catching the cool moonlight with subtle silver highlights. The owl's gaze is curious and full of charm, giving it a whimsical, storybook-like personality."
    negative_prompt: str = "ugly, blurry, low quality"
    seed: int = 42
    # Approximate 16:9 with small width while keeping both dims divisible by 16 (required by FIBO).
    # 320 is 20×16; closest valid 16-multiple height to 320*(9/16)=180 is 176 (11×16).
    # So we use 320×176 ≈ 16:8.8 as a near-16:9 aspect ratio.
    height: int = 176
    width: int = 320
    num_inference_steps: int = 20
    guidance: float = 4.0


TXT2IMG_DEBUG_CONFIG = DebugTxt2ImgConfig()
