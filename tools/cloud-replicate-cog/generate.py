import os
import subprocess
import time
from typing import List, Optional

from cog import BasePredictor, Input, Path

from mflux.models.common.config import Config, ModelConfig
from mflux.models.flux.variants.txt2img.flux import Flux1

MODEL_CACHE = "/data/FLUX.1-schnell"
MODEL_URL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-schnell/files.tar"

ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "21:9": (1536, 640),
    "3:2": (1216, 832),
    "2:3": (832, 1216),
    "4:5": (896, 1088),
    "5:4": (1088, 896),
    "3:4": (896, 1152),
    "4:3": (1152, 896),
    "9:16": (768, 1344),
    "9:21": (640, 1536),
}


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


def find_transformers_parent(start_path: str) -> Optional[Path]:
    """
    Given a starting path, finds the parent directory that contains a 'transformers' directory.

    Args:
        start_path: The path to start the search from.

    Returns:
        The Path object of the parent directory containing 'transformers', or None if not found.
    """
    try:
        # Resolve the path to handle '~' and make it absolute
        p = Path(start_path).expanduser().resolve()

        # Iterate through the path itself and its parents
        for parent in [p] + list(p.parents):
            if (parent / "transformers").is_dir():
                return parent
    except FileNotFoundError:
        return None

    return None


class TextToImage(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        start = time.time()
        print(f"Loading Flux model weights to {MODEL_CACHE}")
        if not Path(MODEL_CACHE).exists():
            download_weights(MODEL_URL, MODEL_CACHE)
        model_path = find_transformers_parent(MODEL_CACHE)
        print(f"model path found at: {model_path}")
        self.flux = Flux1(path=model_path, model="schnell")
        print("setup took: ", time.time() - start)

    def predict(
        self,
        prompt: str = Input(description="Prompt for generated image"),
        steps: int = Input(
            description="Number of inference steps.",
            ge=1,
            le=20,
            default=14,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        seed: Optional[int] = Input(description="Random seed. Set for reproducible generation", default=None),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        guidance = 3.5
        output_paths = []
        for n in range(num_outputs):
            image = self.flux.generate_image(
                seed=seed,
                prompt=prompt,
                num_inference_steps=steps,
                height=512,
                width=512,
                guidance=guidance,
            )
            output_dir = Path("/tmp/mflux-output")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"output-{int(time.time() * 1e6)}.png"
            image.save(output_path)
            output_paths.append(output_path)

        return output_paths
