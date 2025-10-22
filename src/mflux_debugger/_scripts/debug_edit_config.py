from dataclasses import dataclass


@dataclass(frozen=True)
class DebugEditConfig:
    prompt: str = "Change only the dog's fur pattern to Dalmatian: black spots on white background. Preserve identical dog, pose, expression, position, lighting, background, composition, and all other visual elements unchanged."
    negative_prompt: str = "ugly, blurry, low quality, distorted, deformed"
    seed: int = 42
    height: int = 384
    width: int = 512
    num_inference_steps: int = 20  # Full quality generation
    guidance: float = 4.0
    image_path: str = "/Users/filip/Desktop/mflux/dog.jpg"


EDIT_DEBUG_CONFIG = DebugEditConfig()
