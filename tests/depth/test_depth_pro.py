import os
from pathlib import Path

from mflux.models.depth_pro.depth_pro import DepthPro
from mflux.utils.image_compare import ImageCompare


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
            ImageCompare.check_images_close_enough(
                output_image_path,
                reference_image_path,
                "Generated image doesn't match reference image.",
            )

        finally:
            # Cleanup
            if os.path.exists(output_image_path):
                os.remove(output_image_path)
