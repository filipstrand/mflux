from tests.image_generation.helpers.image_generation_upscaler_test_helper import ImageGeneratorUpscalerTestHelper


class TestImageGeneratorUpscaler:
    def test_image_upscaling(self):
        ImageGeneratorUpscalerTestHelper.assert_matches_reference_image(
            reference_image_path="reference_upscaled.png",
            output_image_path="output_upscaler.png",
            input_image_path="low_res.jpg",
            steps=20,
            seed=42,
            height=int(192 * 2),
            width=int(320 * 2),
            prompt="",
            controlnet_strength=0.6,
        )
