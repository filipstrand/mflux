import os
from pathlib import Path

import mlx.core as mx
import pytest

from mflux.models.krea2.variants.controlnet.krea2_controlnet import Krea2Controlnet
from mflux.utils.image_compare import ImageCompare

RESOURCES = Path(__file__).parent.parent / "resources"
# Depth Pro map of reference_flux2_klein_tiger_swim.png (near is white), downscaled to 512x512.
DEPTH = RESOURCES / "krea2_controlnet_depth_tiger.png"
PROMPT = "a bronze statue of a tiger swimming, museum photo"


@pytest.fixture(scope="module")
def model():
    # Krea 2 Turbo at q8 with the depth ControlNet merged in, about 30 GB at its peak.
    model = Krea2Controlnet(quantize=8)
    yield model
    del model
    mx.clear_cache()


@pytest.mark.slow
class TestImageGeneratorKrea2Controlnet:
    def test_depth_control(self, model, tmp_path):
        reference = "reference_krea2_controlnet_depth.png"
        output = tmp_path / f"output_{reference}"
        if "MFLUX_PRESERVE_TEST_OUTPUT" in os.environ:
            output = RESOURCES / f"output_{reference}"
        image = model.generate_image(seed=42, prompt=PROMPT, depth_image_path=str(DEPTH), width=512, height=512)
        image.save(path=output, overwrite=True)
        ImageCompare.check_images_close_enough(
            output,
            RESOURCES / reference,
            "Generated Krea 2 depth ControlNet image doesn't match reference image.",
        )
