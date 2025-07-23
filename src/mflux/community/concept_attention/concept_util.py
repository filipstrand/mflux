import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
import PIL.Image

from mflux.community.concept_attention.attention_data import (
    ConceptHeatmap,
    GenerationAttentionData,
)


class ConceptUtil:
    @staticmethod
    def create_heatmap(
        concept: str,
        attention_data: GenerationAttentionData,
        height: int,
        width: int,
        layer_indices: list[int],
        timesteps: list[int] = list(range(4)),
    ) -> ConceptHeatmap:
        heatmap = ConceptUtil._compute_heatmap(
            attention_data=attention_data,
            height=height,
            width=width,
            layer_indices=layer_indices,
            timesteps=timesteps,
        )

        colorized_image = ConceptUtil._to_heatmap_image(
            heatmap=heatmap,
            height=height,
            width=width,
        )

        return ConceptHeatmap(
            concept=concept,
            image=colorized_image,
            layer_indices=layer_indices,
            timesteps=timesteps,
            height=height,
            width=width,
        )

    @staticmethod
    def create_contrastive_heatmap(
        concept: str,
        attention_data: GenerationAttentionData,
        background_attention_data: GenerationAttentionData,
        height: int,
        width: int,
        layer_indices: list[int],
        timesteps: list[int] = list(range(4)),
        sharpening_exponent: float = 2.0,
        temperature: float = 0.1,
    ) -> ConceptHeatmap:
        """
        Create a contrastive concept heatmap using background as anti-concept.

        Args:
            concept: The target concept (e.g., "dragon")
            attention_data: Attention data for the concept
            background_attention_data: Attention data for "background"
            height, width: Output dimensions
            layer_indices: Which transformer layers to use
            timesteps: Which timesteps to use
            sharpening_exponent: Spectral sharpening power (>1 for sharper)
            temperature: Softmax temperature (<1 for sharper)
        """
        heatmap = ConceptUtil._compute_contrastive_heatmap(
            concept_attention_data=attention_data,
            background_attention_data=background_attention_data,
            height=height,
            width=width,
            layer_indices=layer_indices,
            timesteps=timesteps,
            sharpening_exponent=sharpening_exponent,
            temperature=temperature,
        )

        colorized_image = ConceptUtil._to_heatmap_image(
            heatmap=heatmap,
            height=height,
            width=width,
        )

        return ConceptHeatmap(
            concept=f"{concept} (contrastive)",
            image=colorized_image,
            layer_indices=layer_indices,
            timesteps=timesteps,
            height=height,
            width=width,
        )

    @staticmethod
    def _to_heatmap_image(heatmap: mx.array, height: int, width: int) -> PIL.Image.Image:
        concept_heatmaps_min = heatmap.min()
        concept_heatmaps_max = heatmap.max()
        colored_heatmaps = []
        for heatmap in heatmap:
            heatmap = (heatmap - concept_heatmaps_min) / (concept_heatmaps_max - concept_heatmaps_min + 1e-8)
            colored_heatmap = plt.get_cmap("plasma")(heatmap)
            rgb_image = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)
            colored_heatmaps.append(rgb_image)
        base_image = colored_heatmaps[0]
        scaled_heatmap = ConceptUtil._pixel_perfect_resize(base_image, height, width)
        return scaled_heatmap

    @staticmethod
    def _pixel_perfect_resize(image_array: np.ndarray, target_height: int, target_width: int) -> PIL.Image.Image:
        patch_h, patch_w, channels = image_array.shape
        block_h = target_height // patch_h
        block_w = target_width // patch_w
        scaled_array = np.zeros((target_height, target_width, channels), dtype=np.uint8)
        for i in range(patch_h):
            for j in range(patch_w):
                patch_color = image_array[i, j]
                start_h = i * block_h
                end_h = min(start_h + block_h, target_height)
                start_w = j * block_w
                end_w = min(start_w + block_w, target_width)
                scaled_array[start_h:end_h, start_w:end_w] = patch_color
        return PIL.Image.fromarray(scaled_array)

    @staticmethod
    def _compute_contrastive_heatmap(
        concept_attention_data: GenerationAttentionData,
        background_attention_data: GenerationAttentionData,
        height: int,
        width: int,
        layer_indices: list[int],
        timesteps: list[int] = list(range(4)),
        sharpening_exponent: float = 2.0,
        temperature: float = 0.1,
    ) -> mx.array:
        """
        Core contrastive attention computation with spectral sharpening.
        """
        # Get concept and background attention vectors
        concept_image_vectors = concept_attention_data.stack_all_img_attentions()
        concept_concept_vectors = concept_attention_data.stack_all_concept_attentions()

        background_image_vectors = background_attention_data.stack_all_img_attentions()
        background_concept_vectors = background_attention_data.stack_all_concept_attentions()

        # Compute concept similarities
        concept_similarities = concept_concept_vectors @ mx.transpose(concept_image_vectors, (0, 1, 2, 4, 3))

        # Compute background similarities
        background_similarities = background_concept_vectors @ mx.transpose(background_image_vectors, (0, 1, 2, 4, 3))

        # The key operation: contrastive subtraction
        contrastive_similarities = concept_similarities - background_similarities

        # Apply temperature-scaled softmax for sharper boundaries
        heatmaps = ConceptUtil._temperature_softmax(contrastive_similarities, temperature)

        # Select timesteps and layers
        heatmaps = heatmaps[timesteps]
        heatmaps = heatmaps[:, layer_indices]

        # Average across timesteps and layers
        heatmaps = mx.mean(heatmaps, axis=(0, 1))

        # Reshape to spatial grid
        batch_dim, concept_dim, patch_dim = heatmaps.shape
        heatmaps = mx.reshape(heatmaps, (batch_dim, concept_dim, height // 16, width // 16))
        heatmap = np.array(heatmaps)[0]

        return heatmap

    @staticmethod
    def _temperature_softmax(similarities: mx.array, temperature: float) -> mx.array:
        """
        Apply temperature-scaled softmax for sharper attention.
        """
        scaled_similarities = similarities / temperature
        return mx.softmax(scaled_similarities, axis=-2)

    @staticmethod
    def _compute_heatmap(
        attention_data: GenerationAttentionData,
        height: int,
        width: int,
        layer_indices: list[int],
        timesteps: list[int] = list(range(4)),
    ) -> mx.array:
        image_vectors = attention_data.stack_all_img_attentions()
        concept_vectors = attention_data.stack_all_concept_attentions()
        heatmaps = concept_vectors @ mx.transpose(image_vectors, (0, 1, 2, 4, 3))
        heatmaps = mx.softmax(heatmaps, axis=-2)
        heatmaps = heatmaps[timesteps]
        heatmaps = heatmaps[:, layer_indices]
        heatmaps = mx.mean(heatmaps, axis=(0, 1))
        batch_dim, concept_dim, patch_dim = heatmaps.shape
        heatmaps = mx.reshape(heatmaps, (batch_dim, concept_dim, height // 16, width // 16))
        heatmap = np.array(heatmaps)[0]
        return heatmap
