from dataclasses import dataclass


@dataclass(frozen=True)
class DebugTxt2ImgConfig:
    prompt: str = "A futuristic cityscape at sunset with flying cars and neon lights"
    negative_prompt: str = "ugly, blurry, low quality"
    seed: int = 42
    height: int = 341
    width: int = 768
    num_inference_steps: int = 20
    guidance: float = 4.0


TXT2IMG_DEBUG_CONFIG = DebugTxt2ImgConfig()
