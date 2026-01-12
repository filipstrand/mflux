"""
Chroma LoRA Mapping.

Maps LoRA weight names to Chroma model layer paths.
Key differences from FLUX:
- No norm1.linear targets (Chroma uses pre-computed modulations)
- No norm1_context.linear targets (Chroma uses pre-computed modulations)
- No single_transformer_blocks.norm.linear targets (Chroma uses pre-computed modulations)
"""

from mflux.models.common.lora.mapping.lora_mapping import LoRAMapping, LoRATarget
from mflux.models.common.lora.mapping.lora_transforms import LoraTransforms


class ChromaLoRAMapping(LoRAMapping):
    """
    LoRA mapping for Chroma model.

    Chroma uses the same architecture as FLUX for attention and FFN layers,
    but replaces the modulation linear layers with DistilledGuidanceLayer.
    Therefore, LoRA targets for norm1.linear and norm1_context.linear are excluded.
    """

    @staticmethod
    def get_mapping() -> list[LoRATarget]:
        """Get all LoRA targets for Chroma model."""
        targets = []

        # Standard format targets (mflux format)
        targets.extend(ChromaLoRAMapping._get_standard_transformer_block_targets())
        targets.extend(ChromaLoRAMapping._get_standard_single_transformer_block_targets())

        # BFL/Kohya format targets
        targets.extend(ChromaLoRAMapping._get_bfl_transformer_block_targets())
        targets.extend(ChromaLoRAMapping._get_bfl_single_transformer_block_targets())

        return targets

    @staticmethod
    def _get_standard_transformer_block_targets() -> list[LoRATarget]:
        """Standard format targets for joint transformer blocks (no norm layers)."""
        return [
            # Image attention Q/K/V
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.to_q",
                possible_up_patterns=[
                    "transformer.transformer_blocks.{block}.attn.to_q.lora_B.weight",
                    "transformer.transformer_blocks.{block}.attn.to_q.lora_B",
                    "transformer_blocks.{block}.attn.to_q.lora_up.weight",
                    "transformer_blocks.{block}.attn.to_q.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.transformer_blocks.{block}.attn.to_q.lora_A.weight",
                    "transformer.transformer_blocks.{block}.attn.to_q.lora_A",
                    "transformer_blocks.{block}.attn.to_q.lora_down.weight",
                    "transformer_blocks.{block}.attn.to_q.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.transformer_blocks.{block}.attn.to_q.alpha",
                    "transformer_blocks.{block}.attn.to_q.alpha",
                ],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.to_k",
                possible_up_patterns=[
                    "transformer.transformer_blocks.{block}.attn.to_k.lora_B.weight",
                    "transformer.transformer_blocks.{block}.attn.to_k.lora_B",
                    "transformer_blocks.{block}.attn.to_k.lora_up.weight",
                    "transformer_blocks.{block}.attn.to_k.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.transformer_blocks.{block}.attn.to_k.lora_A.weight",
                    "transformer.transformer_blocks.{block}.attn.to_k.lora_A",
                    "transformer_blocks.{block}.attn.to_k.lora_down.weight",
                    "transformer_blocks.{block}.attn.to_k.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.transformer_blocks.{block}.attn.to_k.alpha",
                    "transformer_blocks.{block}.attn.to_k.alpha",
                ],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.to_v",
                possible_up_patterns=[
                    "transformer.transformer_blocks.{block}.attn.to_v.lora_B.weight",
                    "transformer.transformer_blocks.{block}.attn.to_v.lora_B",
                    "transformer_blocks.{block}.attn.to_v.lora_up.weight",
                    "transformer_blocks.{block}.attn.to_v.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.transformer_blocks.{block}.attn.to_v.lora_A.weight",
                    "transformer.transformer_blocks.{block}.attn.to_v.lora_A",
                    "transformer_blocks.{block}.attn.to_v.lora_down.weight",
                    "transformer_blocks.{block}.attn.to_v.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.transformer_blocks.{block}.attn.to_v.alpha",
                    "transformer_blocks.{block}.attn.to_v.alpha",
                ],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.to_out.0",
                possible_up_patterns=[
                    "transformer.transformer_blocks.{block}.attn.to_out.0.lora_B.weight",
                    "transformer.transformer_blocks.{block}.attn.to_out.0.lora_B",
                    "transformer_blocks.{block}.attn.to_out.0.lora_up.weight",
                    "transformer_blocks.{block}.attn.to_out.0.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.transformer_blocks.{block}.attn.to_out.0.lora_A.weight",
                    "transformer.transformer_blocks.{block}.attn.to_out.0.lora_A",
                    "transformer_blocks.{block}.attn.to_out.0.lora_down.weight",
                    "transformer_blocks.{block}.attn.to_out.0.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.transformer_blocks.{block}.attn.to_out.0.alpha",
                    "transformer_blocks.{block}.attn.to_out.0.alpha",
                ],
            ),
            # Text/context attention Q/K/V
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.add_q_proj",
                possible_up_patterns=[
                    "transformer.transformer_blocks.{block}.attn.add_q_proj.lora_B.weight",
                    "transformer.transformer_blocks.{block}.attn.add_q_proj.lora_B",
                    "transformer_blocks.{block}.attn.add_q_proj.lora_up.weight",
                    "transformer_blocks.{block}.attn.add_q_proj.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.transformer_blocks.{block}.attn.add_q_proj.lora_A.weight",
                    "transformer.transformer_blocks.{block}.attn.add_q_proj.lora_A",
                    "transformer_blocks.{block}.attn.add_q_proj.lora_down.weight",
                    "transformer_blocks.{block}.attn.add_q_proj.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.transformer_blocks.{block}.attn.add_q_proj.alpha",
                    "transformer_blocks.{block}.attn.add_q_proj.alpha",
                ],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.add_k_proj",
                possible_up_patterns=[
                    "transformer.transformer_blocks.{block}.attn.add_k_proj.lora_B.weight",
                    "transformer.transformer_blocks.{block}.attn.add_k_proj.lora_B",
                    "transformer_blocks.{block}.attn.add_k_proj.lora_up.weight",
                    "transformer_blocks.{block}.attn.add_k_proj.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.transformer_blocks.{block}.attn.add_k_proj.lora_A.weight",
                    "transformer.transformer_blocks.{block}.attn.add_k_proj.lora_A",
                    "transformer_blocks.{block}.attn.add_k_proj.lora_down.weight",
                    "transformer_blocks.{block}.attn.add_k_proj.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.transformer_blocks.{block}.attn.add_k_proj.alpha",
                    "transformer_blocks.{block}.attn.add_k_proj.alpha",
                ],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.add_v_proj",
                possible_up_patterns=[
                    "transformer.transformer_blocks.{block}.attn.add_v_proj.lora_B.weight",
                    "transformer.transformer_blocks.{block}.attn.add_v_proj.lora_B",
                    "transformer_blocks.{block}.attn.add_v_proj.lora_up.weight",
                    "transformer_blocks.{block}.attn.add_v_proj.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.transformer_blocks.{block}.attn.add_v_proj.lora_A.weight",
                    "transformer.transformer_blocks.{block}.attn.add_v_proj.lora_A",
                    "transformer_blocks.{block}.attn.add_v_proj.lora_down.weight",
                    "transformer_blocks.{block}.attn.add_v_proj.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.transformer_blocks.{block}.attn.add_v_proj.alpha",
                    "transformer_blocks.{block}.attn.add_v_proj.alpha",
                ],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.to_add_out",
                possible_up_patterns=[
                    "transformer.transformer_blocks.{block}.attn.to_add_out.lora_B.weight",
                    "transformer.transformer_blocks.{block}.attn.to_add_out.lora_B",
                    "transformer_blocks.{block}.attn.to_add_out.lora_up.weight",
                    "transformer_blocks.{block}.attn.to_add_out.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.transformer_blocks.{block}.attn.to_add_out.lora_A.weight",
                    "transformer.transformer_blocks.{block}.attn.to_add_out.lora_A",
                    "transformer_blocks.{block}.attn.to_add_out.lora_down.weight",
                    "transformer_blocks.{block}.attn.to_add_out.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.transformer_blocks.{block}.attn.to_add_out.alpha",
                    "transformer_blocks.{block}.attn.to_add_out.alpha",
                ],
            ),
            # Image feed-forward
            LoRATarget(
                model_path="transformer_blocks.{block}.ff.linear1",
                possible_up_patterns=[
                    "transformer.transformer_blocks.{block}.ff.net.0.proj.lora_B.weight",
                    "transformer.transformer_blocks.{block}.ff.net.0.proj.lora_B",
                    "transformer.transformer_blocks.{block}.ff.linear1.lora_B.weight",
                    "transformer.transformer_blocks.{block}.ff.linear1.lora_B",
                    "transformer_blocks.{block}.ff.net.0.proj.lora_up.weight",
                    "transformer_blocks.{block}.ff.linear1.lora_up.weight",
                ],
                possible_down_patterns=[
                    "transformer.transformer_blocks.{block}.ff.net.0.proj.lora_A.weight",
                    "transformer.transformer_blocks.{block}.ff.net.0.proj.lora_A",
                    "transformer.transformer_blocks.{block}.ff.linear1.lora_A.weight",
                    "transformer.transformer_blocks.{block}.ff.linear1.lora_A",
                    "transformer_blocks.{block}.ff.net.0.proj.lora_down.weight",
                    "transformer_blocks.{block}.ff.linear1.lora_down.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.transformer_blocks.{block}.ff.net.0.proj.alpha",
                    "transformer.transformer_blocks.{block}.ff.linear1.alpha",
                    "transformer_blocks.{block}.ff.net.0.proj.alpha",
                ],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.ff.linear2",
                possible_up_patterns=[
                    "transformer.transformer_blocks.{block}.ff.net.2.lora_B.weight",
                    "transformer.transformer_blocks.{block}.ff.net.2.lora_B",
                    "transformer.transformer_blocks.{block}.ff.linear2.lora_B.weight",
                    "transformer.transformer_blocks.{block}.ff.linear2.lora_B",
                    "transformer_blocks.{block}.ff.net.2.lora_up.weight",
                    "transformer_blocks.{block}.ff.linear2.lora_up.weight",
                ],
                possible_down_patterns=[
                    "transformer.transformer_blocks.{block}.ff.net.2.lora_A.weight",
                    "transformer.transformer_blocks.{block}.ff.net.2.lora_A",
                    "transformer.transformer_blocks.{block}.ff.linear2.lora_A.weight",
                    "transformer.transformer_blocks.{block}.ff.linear2.lora_A",
                    "transformer_blocks.{block}.ff.net.2.lora_down.weight",
                    "transformer_blocks.{block}.ff.linear2.lora_down.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.transformer_blocks.{block}.ff.net.2.alpha",
                    "transformer.transformer_blocks.{block}.ff.linear2.alpha",
                    "transformer_blocks.{block}.ff.net.2.alpha",
                ],
            ),
            # Context/text feed-forward
            LoRATarget(
                model_path="transformer_blocks.{block}.ff_context.linear1",
                possible_up_patterns=[
                    "transformer.transformer_blocks.{block}.ff_context.net.0.proj.lora_B.weight",
                    "transformer.transformer_blocks.{block}.ff_context.net.0.proj.lora_B",
                    "transformer.transformer_blocks.{block}.ff_context.linear1.lora_B.weight",
                    "transformer.transformer_blocks.{block}.ff_context.linear1.lora_B",
                    "transformer_blocks.{block}.ff_context.net.0.proj.lora_up.weight",
                    "transformer_blocks.{block}.ff_context.linear1.lora_up.weight",
                ],
                possible_down_patterns=[
                    "transformer.transformer_blocks.{block}.ff_context.net.0.proj.lora_A.weight",
                    "transformer.transformer_blocks.{block}.ff_context.net.0.proj.lora_A",
                    "transformer.transformer_blocks.{block}.ff_context.linear1.lora_A.weight",
                    "transformer.transformer_blocks.{block}.ff_context.linear1.lora_A",
                    "transformer_blocks.{block}.ff_context.net.0.proj.lora_down.weight",
                    "transformer_blocks.{block}.ff_context.linear1.lora_down.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.transformer_blocks.{block}.ff_context.net.0.proj.alpha",
                    "transformer.transformer_blocks.{block}.ff_context.linear1.alpha",
                    "transformer_blocks.{block}.ff_context.net.0.proj.alpha",
                ],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.ff_context.linear2",
                possible_up_patterns=[
                    "transformer.transformer_blocks.{block}.ff_context.net.2.lora_B.weight",
                    "transformer.transformer_blocks.{block}.ff_context.net.2.lora_B",
                    "transformer.transformer_blocks.{block}.ff_context.linear2.lora_B.weight",
                    "transformer.transformer_blocks.{block}.ff_context.linear2.lora_B",
                    "transformer_blocks.{block}.ff_context.net.2.lora_up.weight",
                    "transformer_blocks.{block}.ff_context.linear2.lora_up.weight",
                ],
                possible_down_patterns=[
                    "transformer.transformer_blocks.{block}.ff_context.net.2.lora_A.weight",
                    "transformer.transformer_blocks.{block}.ff_context.net.2.lora_A",
                    "transformer.transformer_blocks.{block}.ff_context.linear2.lora_A.weight",
                    "transformer.transformer_blocks.{block}.ff_context.linear2.lora_A",
                    "transformer_blocks.{block}.ff_context.net.2.lora_down.weight",
                    "transformer_blocks.{block}.ff_context.linear2.lora_down.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.transformer_blocks.{block}.ff_context.net.2.alpha",
                    "transformer.transformer_blocks.{block}.ff_context.linear2.alpha",
                    "transformer_blocks.{block}.ff_context.net.2.alpha",
                ],
            ),
            # NOTE: No norm1.linear or norm1_context.linear targets
            # Chroma uses DistilledGuidanceLayer instead of per-block modulation
        ]

    @staticmethod
    def _get_standard_single_transformer_block_targets() -> list[LoRATarget]:
        """Standard format targets for single transformer blocks (no norm layers)."""
        return [
            LoRATarget(
                model_path="single_transformer_blocks.{block}.attn.to_q",
                possible_up_patterns=[
                    "transformer.single_transformer_blocks.{block}.attn.to_q.lora_B.weight",
                    "transformer.single_transformer_blocks.{block}.attn.to_q.lora_B",
                    "single_transformer_blocks.{block}.attn.to_q.lora_up.weight",
                    "single_transformer_blocks.{block}.attn.to_q.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.single_transformer_blocks.{block}.attn.to_q.lora_A.weight",
                    "transformer.single_transformer_blocks.{block}.attn.to_q.lora_A",
                    "single_transformer_blocks.{block}.attn.to_q.lora_down.weight",
                    "single_transformer_blocks.{block}.attn.to_q.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.single_transformer_blocks.{block}.attn.to_q.alpha",
                    "single_transformer_blocks.{block}.attn.to_q.alpha",
                ],
            ),
            LoRATarget(
                model_path="single_transformer_blocks.{block}.attn.to_k",
                possible_up_patterns=[
                    "transformer.single_transformer_blocks.{block}.attn.to_k.lora_B.weight",
                    "transformer.single_transformer_blocks.{block}.attn.to_k.lora_B",
                    "single_transformer_blocks.{block}.attn.to_k.lora_up.weight",
                    "single_transformer_blocks.{block}.attn.to_k.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.single_transformer_blocks.{block}.attn.to_k.lora_A.weight",
                    "transformer.single_transformer_blocks.{block}.attn.to_k.lora_A",
                    "single_transformer_blocks.{block}.attn.to_k.lora_down.weight",
                    "single_transformer_blocks.{block}.attn.to_k.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.single_transformer_blocks.{block}.attn.to_k.alpha",
                    "single_transformer_blocks.{block}.attn.to_k.alpha",
                ],
            ),
            LoRATarget(
                model_path="single_transformer_blocks.{block}.attn.to_v",
                possible_up_patterns=[
                    "transformer.single_transformer_blocks.{block}.attn.to_v.lora_B.weight",
                    "transformer.single_transformer_blocks.{block}.attn.to_v.lora_B",
                    "single_transformer_blocks.{block}.attn.to_v.lora_up.weight",
                    "single_transformer_blocks.{block}.attn.to_v.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.single_transformer_blocks.{block}.attn.to_v.lora_A.weight",
                    "transformer.single_transformer_blocks.{block}.attn.to_v.lora_A",
                    "single_transformer_blocks.{block}.attn.to_v.lora_down.weight",
                    "single_transformer_blocks.{block}.attn.to_v.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.single_transformer_blocks.{block}.attn.to_v.alpha",
                    "single_transformer_blocks.{block}.attn.to_v.alpha",
                ],
            ),
            LoRATarget(
                model_path="single_transformer_blocks.{block}.proj_mlp",
                possible_up_patterns=[
                    "transformer.single_transformer_blocks.{block}.proj_mlp.lora_B.weight",
                    "transformer.single_transformer_blocks.{block}.proj_mlp.lora_B",
                    "single_transformer_blocks.{block}.proj_mlp.lora_up.weight",
                    "single_transformer_blocks.{block}.proj_mlp.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.single_transformer_blocks.{block}.proj_mlp.lora_A.weight",
                    "transformer.single_transformer_blocks.{block}.proj_mlp.lora_A",
                    "single_transformer_blocks.{block}.proj_mlp.lora_down.weight",
                    "single_transformer_blocks.{block}.proj_mlp.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.single_transformer_blocks.{block}.proj_mlp.alpha",
                    "single_transformer_blocks.{block}.proj_mlp.alpha",
                ],
            ),
            LoRATarget(
                model_path="single_transformer_blocks.{block}.proj_out",
                possible_up_patterns=[
                    "transformer.single_transformer_blocks.{block}.proj_out.lora_B.weight",
                    "transformer.single_transformer_blocks.{block}.proj_out.lora_B",
                    "single_transformer_blocks.{block}.proj_out.lora_up.weight",
                    "single_transformer_blocks.{block}.proj_out.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.single_transformer_blocks.{block}.proj_out.lora_A.weight",
                    "transformer.single_transformer_blocks.{block}.proj_out.lora_A",
                    "single_transformer_blocks.{block}.proj_out.lora_down.weight",
                    "single_transformer_blocks.{block}.proj_out.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.single_transformer_blocks.{block}.proj_out.alpha",
                    "single_transformer_blocks.{block}.proj_out.alpha",
                ],
            ),
            # NOTE: No norm.linear targets
            # Chroma uses DistilledGuidanceLayer instead of per-block modulation
        ]

    @staticmethod
    def _get_bfl_transformer_block_targets() -> list[LoRATarget]:
        """BFL/Kohya format targets for joint transformer blocks (no norm layers)."""
        return [
            # Image attention with QKV split transforms
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.to_q",
                possible_up_patterns=["lora_unet_double_blocks_{block}_img_attn_qkv.lora_up.weight"],
                possible_down_patterns=["lora_unet_double_blocks_{block}_img_attn_qkv.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_double_blocks_{block}_img_attn_qkv.alpha"],
                up_transform=LoraTransforms.split_q_up,
                down_transform=LoraTransforms.split_q_down,
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.to_k",
                possible_up_patterns=["lora_unet_double_blocks_{block}_img_attn_qkv.lora_up.weight"],
                possible_down_patterns=["lora_unet_double_blocks_{block}_img_attn_qkv.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_double_blocks_{block}_img_attn_qkv.alpha"],
                up_transform=LoraTransforms.split_k_up,
                down_transform=LoraTransforms.split_k_down,
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.to_v",
                possible_up_patterns=["lora_unet_double_blocks_{block}_img_attn_qkv.lora_up.weight"],
                possible_down_patterns=["lora_unet_double_blocks_{block}_img_attn_qkv.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_double_blocks_{block}_img_attn_qkv.alpha"],
                up_transform=LoraTransforms.split_v_up,
                down_transform=LoraTransforms.split_v_down,
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.to_out.0",
                possible_up_patterns=["lora_unet_double_blocks_{block}_img_attn_proj.lora_up.weight"],
                possible_down_patterns=["lora_unet_double_blocks_{block}_img_attn_proj.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_double_blocks_{block}_img_attn_proj.alpha"],
            ),
            # Text attention with QKV split transforms
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.add_q_proj",
                possible_up_patterns=["lora_unet_double_blocks_{block}_txt_attn_qkv.lora_up.weight"],
                possible_down_patterns=["lora_unet_double_blocks_{block}_txt_attn_qkv.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_double_blocks_{block}_txt_attn_qkv.alpha"],
                up_transform=LoraTransforms.split_q_up,
                down_transform=LoraTransforms.split_q_down,
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.add_k_proj",
                possible_up_patterns=["lora_unet_double_blocks_{block}_txt_attn_qkv.lora_up.weight"],
                possible_down_patterns=["lora_unet_double_blocks_{block}_txt_attn_qkv.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_double_blocks_{block}_txt_attn_qkv.alpha"],
                up_transform=LoraTransforms.split_k_up,
                down_transform=LoraTransforms.split_k_down,
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.add_v_proj",
                possible_up_patterns=["lora_unet_double_blocks_{block}_txt_attn_qkv.lora_up.weight"],
                possible_down_patterns=["lora_unet_double_blocks_{block}_txt_attn_qkv.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_double_blocks_{block}_txt_attn_qkv.alpha"],
                up_transform=LoraTransforms.split_v_up,
                down_transform=LoraTransforms.split_v_down,
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.to_add_out",
                possible_up_patterns=["lora_unet_double_blocks_{block}_txt_attn_proj.lora_up.weight"],
                possible_down_patterns=["lora_unet_double_blocks_{block}_txt_attn_proj.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_double_blocks_{block}_txt_attn_proj.alpha"],
            ),
            # Image feed-forward
            LoRATarget(
                model_path="transformer_blocks.{block}.ff.linear1",
                possible_up_patterns=["lora_unet_double_blocks_{block}_img_mlp_0.lora_up.weight"],
                possible_down_patterns=["lora_unet_double_blocks_{block}_img_mlp_0.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_double_blocks_{block}_img_mlp_0.alpha"],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.ff.linear2",
                possible_up_patterns=["lora_unet_double_blocks_{block}_img_mlp_2.lora_up.weight"],
                possible_down_patterns=["lora_unet_double_blocks_{block}_img_mlp_2.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_double_blocks_{block}_img_mlp_2.alpha"],
            ),
            # Text feed-forward
            LoRATarget(
                model_path="transformer_blocks.{block}.ff_context.linear1",
                possible_up_patterns=["lora_unet_double_blocks_{block}_txt_mlp_0.lora_up.weight"],
                possible_down_patterns=["lora_unet_double_blocks_{block}_txt_mlp_0.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_double_blocks_{block}_txt_mlp_0.alpha"],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.ff_context.linear2",
                possible_up_patterns=["lora_unet_double_blocks_{block}_txt_mlp_2.lora_up.weight"],
                possible_down_patterns=["lora_unet_double_blocks_{block}_txt_mlp_2.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_double_blocks_{block}_txt_mlp_2.alpha"],
            ),
            # NOTE: No img_mod_lin or txt_mod_lin targets
            # Chroma uses DistilledGuidanceLayer instead of per-block modulation
        ]

    @staticmethod
    def _get_bfl_single_transformer_block_targets() -> list[LoRATarget]:
        """BFL/Kohya format targets for single transformer blocks (no norm layers)."""
        return [
            # Attention Q/K/V + MLP from combined linear1
            LoRATarget(
                model_path="single_transformer_blocks.{block}.attn.to_q",
                possible_up_patterns=["lora_unet_single_blocks_{block}_linear1.lora_up.weight"],
                possible_down_patterns=["lora_unet_single_blocks_{block}_linear1.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_single_blocks_{block}_linear1.alpha"],
                up_transform=LoraTransforms.split_single_q_up,
                down_transform=LoraTransforms.split_single_q_down,
            ),
            LoRATarget(
                model_path="single_transformer_blocks.{block}.attn.to_k",
                possible_up_patterns=["lora_unet_single_blocks_{block}_linear1.lora_up.weight"],
                possible_down_patterns=["lora_unet_single_blocks_{block}_linear1.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_single_blocks_{block}_linear1.alpha"],
                up_transform=LoraTransforms.split_single_k_up,
                down_transform=LoraTransforms.split_single_k_down,
            ),
            LoRATarget(
                model_path="single_transformer_blocks.{block}.attn.to_v",
                possible_up_patterns=["lora_unet_single_blocks_{block}_linear1.lora_up.weight"],
                possible_down_patterns=["lora_unet_single_blocks_{block}_linear1.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_single_blocks_{block}_linear1.alpha"],
                up_transform=LoraTransforms.split_single_v_up,
                down_transform=LoraTransforms.split_single_v_down,
            ),
            LoRATarget(
                model_path="single_transformer_blocks.{block}.proj_mlp",
                possible_up_patterns=["lora_unet_single_blocks_{block}_linear1.lora_up.weight"],
                possible_down_patterns=["lora_unet_single_blocks_{block}_linear1.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_single_blocks_{block}_linear1.alpha"],
                up_transform=LoraTransforms.split_single_mlp_up,
                down_transform=LoraTransforms.split_single_mlp_down,
            ),
            # Output projection from linear2
            LoRATarget(
                model_path="single_transformer_blocks.{block}.proj_out",
                possible_up_patterns=["lora_unet_single_blocks_{block}_linear2.lora_up.weight"],
                possible_down_patterns=["lora_unet_single_blocks_{block}_linear2.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_single_blocks_{block}_linear2.alpha"],
            ),
            # NOTE: No modulation_lin targets
            # Chroma uses DistilledGuidanceLayer instead of per-block modulation
        ]
