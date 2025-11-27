from dataclasses import dataclass
from pathlib import Path
from typing import List

import mlx.core as mx
import PIL.Image

from mflux.models.flux.variants.concept_attention.joint_transformer_block_concept import LayerAttentionData


@dataclass
class TimestepAttentionData:
    t: int
    attention_information: List[LayerAttentionData]

    def stack_img_attentions(self) -> mx.array:
        return mx.stack([layer.img_attention for layer in self.attention_information], axis=0)

    def stack_concept_attentions(self) -> mx.array:
        return mx.stack([layer.concept_attention for layer in self.attention_information], axis=0)


class GenerationAttentionData:
    def __init__(self):
        self.timestep_data: List[TimestepAttentionData] = []

    def append(self, timestep_attention: TimestepAttentionData):
        self.timestep_data.append(timestep_attention)

    def stack_all_img_attentions(self) -> mx.array:
        timestep_stacks = [timestep.stack_img_attentions() for timestep in self.timestep_data]
        return mx.stack(timestep_stacks, axis=0)

    def stack_all_concept_attentions(self) -> mx.array:
        timestep_stacks = [timestep.stack_concept_attentions() for timestep in self.timestep_data]
        return mx.stack(timestep_stacks, axis=0)


@dataclass
class ConceptHeatmap:
    concept: str
    image: PIL.Image.Image
    layer_indices: List[int]
    timesteps: List[int]
    height: int
    width: int

    def save(self, path: str | Path, export_json_metadata: bool = False, overwrite: bool = False) -> None:
        from mflux.utils.image_util import ImageUtil

        ImageUtil.save_image(
            image=self.image,
            path=path,
            metadata=self.get_metadata(),
            export_json_metadata=export_json_metadata,
            overwrite=overwrite,
        )

    def get_metadata(self) -> dict:
        return {
            "concept_prompt": self.concept,
            "layer_indices": self.layer_indices,
            "timesteps": self.timesteps,
            "height": self.height,
            "width": self.width,
        }
