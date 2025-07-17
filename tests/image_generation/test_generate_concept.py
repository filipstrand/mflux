from mflux.config.model_config import ModelConfig
from tests.image_generation.helpers.image_generation_concept_test_helper import ImageGenerationConceptTestHelper


class TestImageGeneratorConcept:
    def test_concept_attention_generation(self):
        ImageGenerationConceptTestHelper.assert_matches_reference_image_concept(
            reference_heatmap_path="reference_concept_schnell_heatmap.png",
            output_heatmap_path="output_concept_schnell_heatmap.png",
            model_config=ModelConfig.schnell(),
            prompt="A dragon on a hill",
            concept="dragon",
            steps=4,
            seed=44,
            height=512,
            width=512,
            heatmap_layer_indices=[15, 16, 17, 18],
            heatmap_timesteps=[0, 1, 2, 3],
        )

    def test_concept_attention_from_image(self):
        ImageGenerationConceptTestHelper.assert_matches_reference_image_concept_from_image(
            reference_heatmap_path="reference_concept_from_image_schnell_heatmap.png",
            output_heatmap_path="output_concept_from_image_schnell_heatmap.png",
            input_image_path="reference_depth_dev_from_image.png",
            model_config=ModelConfig.schnell(),
            prompt="A photo of cartoon of Albert Einstein",
            concept="man",
            steps=4,
            seed=42,
            height=512,
            width=320,
            heatmap_layer_indices=[15, 16, 17, 18],
            heatmap_timesteps=[0, 1, 2, 3],
        )
