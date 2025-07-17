import os
from pathlib import Path

import numpy as np
from PIL import Image

from mflux.community.concept_attention.flux_concept import Flux1Concept
from mflux.community.concept_attention.flux_concept_from_image import Flux1ConceptFromImage
from mflux.config.config import Config
from mflux.config.model_config import ModelConfig


class ImageGenerationConceptTestHelper:
    @staticmethod
    def assert_matches_reference_image_concept(
        reference_heatmap_path: str,
        output_heatmap_path: str,
        model_config: ModelConfig,
        prompt: str,
        concept: str,
        steps: int,
        seed: int,
        height: int | None = None,
        width: int | None = None,
        heatmap_layer_indices: list[int] | None = None,
        heatmap_timesteps: list[int] | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ):
        # resolve paths
        reference_heatmap_path = ImageGenerationConceptTestHelper.resolve_path(reference_heatmap_path)
        output_heatmap_path = ImageGenerationConceptTestHelper.resolve_path(output_heatmap_path)
        lora_paths = [str(ImageGenerationConceptTestHelper.resolve_path(p)) for p in lora_paths] if lora_paths else None

        try:
            # given
            flux = Flux1Concept(
                model_config=model_config,
                quantize=4,
                lora_paths=lora_paths,
                lora_scales=lora_scales,
            )
            # when
            image = flux.generate_image(
                seed=seed,
                prompt=prompt,
                concept=concept,
                heatmap_layer_indices=heatmap_layer_indices,
                heatmap_timesteps=heatmap_timesteps,
                config=Config(
                    num_inference_steps=steps,
                    height=height,
                    width=width,
                ),
            )
            # Save only the heatmap (we don't need the original image for testing)
            image.save_concept_heatmap(path=output_heatmap_path, overwrite=True)

            # then - verify the heatmap matches reference
            np.testing.assert_array_equal(
                np.array(Image.open(output_heatmap_path)),
                np.array(Image.open(reference_heatmap_path)),
                err_msg=f"Generated concept heatmap doesn't match reference heatmap. Check {output_heatmap_path} vs {reference_heatmap_path}",
            )

        finally:
            # cleanup
            if os.path.exists(output_heatmap_path):
                os.remove(output_heatmap_path)

    @staticmethod
    def assert_matches_reference_image_concept_from_image(
        reference_heatmap_path: str,
        output_heatmap_path: str,
        input_image_path: str,
        model_config: ModelConfig,
        prompt: str,
        concept: str,
        steps: int,
        seed: int,
        height: int | None = None,
        width: int | None = None,
        heatmap_layer_indices: list[int] | None = None,
        heatmap_timesteps: list[int] | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ):
        # resolve paths
        reference_heatmap_path = ImageGenerationConceptTestHelper.resolve_path(reference_heatmap_path)
        output_heatmap_path = ImageGenerationConceptTestHelper.resolve_path(output_heatmap_path)
        input_image_path = ImageGenerationConceptTestHelper.resolve_path(input_image_path)
        lora_paths = [str(ImageGenerationConceptTestHelper.resolve_path(p)) for p in lora_paths] if lora_paths else None

        try:
            # given
            flux = Flux1ConceptFromImage(
                model_config=model_config,
                quantize=8,
                lora_paths=lora_paths,
                lora_scales=lora_scales,
            )
            # when
            image = flux.generate_image(
                seed=seed,
                prompt=prompt,
                concept=concept,
                image_path=str(input_image_path),
                heatmap_layer_indices=heatmap_layer_indices,
                heatmap_timesteps=heatmap_timesteps,
                config=Config(
                    num_inference_steps=steps,
                    height=height,
                    width=width,
                ),
            )
            # Save only the heatmap (we don't need the original image for testing)
            image.save_concept_heatmap(path=output_heatmap_path, overwrite=True)

            # then - verify the heatmap matches reference
            np.testing.assert_array_equal(
                np.array(Image.open(output_heatmap_path)),
                np.array(Image.open(reference_heatmap_path)),
                err_msg=f"Generated concept from image heatmap doesn't match reference heatmap. Check {output_heatmap_path} vs {reference_heatmap_path}",
            )

        finally:
            # cleanup
            if os.path.exists(output_heatmap_path):
                os.remove(output_heatmap_path)

    @staticmethod
    def resolve_path(path) -> Path | None:
        if path is None:
            return None
        return Path(__file__).parent.parent.parent / "resources" / path
