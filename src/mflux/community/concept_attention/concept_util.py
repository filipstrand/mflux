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
    def _to_heatmap_image(heatmap: mx.array, height: int, width: int) -> PIL.Image.Image:
        concept_heatmaps_min = heatmap.min()
        concept_heatmaps_max = heatmap.max()
        colored_heatmaps = []
        for heatmap in heatmap:
            heatmap = (heatmap - concept_heatmaps_min) / (
                concept_heatmaps_max - concept_heatmaps_min + 1e-8  # Add small epsilon to prevent division by zero
            )
            colored_heatmap = plt.get_cmap("plasma")(heatmap)
            rgb_image = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)
            colored_heatmaps.append(rgb_image)

        heatmap = [PIL.Image.fromarray(concept_heatmap) for concept_heatmap in colored_heatmaps]
        scaled_heatmap = heatmap[0].resize((width, height), PIL.Image.LANCZOS)
        return scaled_heatmap

    @staticmethod
    def _compute_heatmap(
        attention_data: GenerationAttentionData,
        layer_indices: list[int],
        timesteps: list[int] = list(range(4)),
    ) -> mx.array:
        image_vectors = attention_data.stack_all_img_attentions()
        concept_vectors = attention_data.stack_all_concept_attentions()
        heatmaps = mx.einsum("tlbpd,tlbcd->tlbcp", image_vectors, concept_vectors)
        heatmaps = mx.softmax(heatmaps, axis=-2)
        heatmaps = heatmaps[timesteps]
        heatmaps = heatmaps[:, layer_indices]
        heatmaps = mx.mean(heatmaps, axis=(0, 1))
        num_patches = heatmaps.shape[-1]
        patch_side = int(num_patches**0.5)
        batch_dim, concept_dim, patch_dim = heatmaps.shape
        heatmaps = mx.reshape(heatmaps, (batch_dim, concept_dim, patch_side, patch_side))
        heatmap = np.array(heatmaps)[0]
        return heatmap
