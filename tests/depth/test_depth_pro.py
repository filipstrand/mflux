import os
import platform
from pathlib import Path

import pytest

from mflux.models.depth_pro.model.depth_pro import DepthPro
from mflux.utils.image_compare import ImageCompare


class TestDepthPro:
    @pytest.mark.skipif(platform.system() == "Linux", reason="Linux-specific MLX core library failure")
    @pytest.mark.slow
    def test_depth_pro_generation(self):
        # Resolve paths
        resource_dir = Path(__file__).parent.parent / "resources"
        input_image_path = resource_dir / "reference_controlnet_dev_lora.png"
        reference_image_path = resource_dir / "reference_depth.png"
        output_image_path = resource_dir / "output_depth.png"

        try:
            # Initialize DepthPro model
            depth_pro = DepthPro()

            # Process the image to generate a depth map
            depth_result = depth_pro.create_depth_map(str(input_image_path))

            # Save the output image
            depth_result.depth_image.save(output_image_path)

            # Assert that the generated depth image matches the reference image
            ImageCompare.check_images_close_enough(
                output_image_path,
                reference_image_path,
                "Generated image doesn't match reference image.",
            )

        finally:
            # Cleanup
            if os.path.exists(output_image_path) and "MFLUX_PRESERVE_TEST_OUTPUT" not in os.environ:
                os.remove(output_image_path)
