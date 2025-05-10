import os
from pathlib import Path

import numpy as np
from PIL import Image

from mflux.models.depth_pro.depth_pro import DepthPro


class TestDepthPro:
    def test_depth_pro_generation(self):
        # Resolve paths
        resource_dir = Path(__file__).parent.parent / "resources"
        input_image_path = resource_dir / "reference_controlnet_dev_lora.png"
        reference_image_path = resource_dir / "reference_controlnet_dev_lora_depth.png"
        output_image_path = resource_dir / "depth_output.png"

        try:
            # Initialize DepthPro model
            depth_pro = DepthPro()

            # Process the image to generate a depth map
            depth_result = depth_pro.create_depth_map(str(input_image_path))

            # Save the output image
            depth_result.depth_image.save(output_image_path)

            # Assert that the generated depth image matches the reference image
            np.testing.assert_array_almost_equal(
                np.array(Image.open(output_image_path)),
                np.array(Image.open(reference_image_path)),
                err_msg=f"Generated depth image doesn't match reference depth image. Check {output_image_path} vs {reference_image_path}",
            )

        finally:
            # Cleanup
            if os.path.exists(output_image_path):
                os.remove(output_image_path)
