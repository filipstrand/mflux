"""LoRA mapping for FLUX.2 model.

Maps LoRA weight patterns to FLUX.2 model paths.

Key differences from FLUX.1:
- 8 joint blocks (vs 19) and 48 single blocks (vs 38)
- FF layers use linear_in/linear_out instead of linear1/linear2
- Single blocks use fused to_qkv_mlp_proj instead of separate projections
- Global modulation layers (not per-block)
"""

from mflux.models.common.lora.mapping.lora_mapping import LoRAMapping, LoRATarget
from mflux.models.common.lora.mapping.lora_transforms import LoraTransforms


class Flux2LoRAMapping(LoRAMapping):
    """LoRA mapping for FLUX.2 transformer."""

    @staticmethod
    def get_mapping() -> list[LoRATarget]:
        targets = []

        targets.extend(Flux2LoRAMapping._get_standard_transformer_block_targets())
        targets.extend(Flux2LoRAMapping._get_standard_single_transformer_block_targets())

        targets.extend(Flux2LoRAMapping._get_bfl_transformer_block_targets())
        targets.extend(Flux2LoRAMapping._get_bfl_single_transformer_block_targets())

        targets.extend(Flux2LoRAMapping._get_global_modulation_targets())

        return targets

    @staticmethod
    def _get_standard_transformer_block_targets() -> list[LoRATarget]:
        """Standard LoRA patterns for joint transformer blocks (8 blocks)."""
        return [
            # Image stream attention
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
            # Context/text stream attention
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
            # Image stream feed-forward (uses linear_in/linear_out for FLUX.2)
            LoRATarget(
                model_path="transformer_blocks.{block}.ff.linear_in",
                possible_up_patterns=[
                    "transformer.transformer_blocks.{block}.ff.linear_in.lora_B.weight",
                    "transformer.transformer_blocks.{block}.ff.linear_in.lora_B",
                    "transformer.transformer_blocks.{block}.ff.net.0.proj.lora_B.weight",
                    "transformer_blocks.{block}.ff.linear_in.lora_up.weight",
                    "transformer_blocks.{block}.ff.linear_in.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.transformer_blocks.{block}.ff.linear_in.lora_A.weight",
                    "transformer.transformer_blocks.{block}.ff.linear_in.lora_A",
                    "transformer.transformer_blocks.{block}.ff.net.0.proj.lora_A.weight",
                    "transformer_blocks.{block}.ff.linear_in.lora_down.weight",
                    "transformer_blocks.{block}.ff.linear_in.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.transformer_blocks.{block}.ff.linear_in.alpha",
                    "transformer.transformer_blocks.{block}.ff.net.0.proj.alpha",
                    "transformer_blocks.{block}.ff.linear_in.alpha",
                ],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.ff.linear_out",
                possible_up_patterns=[
                    "transformer.transformer_blocks.{block}.ff.linear_out.lora_B.weight",
                    "transformer.transformer_blocks.{block}.ff.linear_out.lora_B",
                    "transformer.transformer_blocks.{block}.ff.net.2.lora_B.weight",
                    "transformer_blocks.{block}.ff.linear_out.lora_up.weight",
                    "transformer_blocks.{block}.ff.linear_out.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.transformer_blocks.{block}.ff.linear_out.lora_A.weight",
                    "transformer.transformer_blocks.{block}.ff.linear_out.lora_A",
                    "transformer.transformer_blocks.{block}.ff.net.2.lora_A.weight",
                    "transformer_blocks.{block}.ff.linear_out.lora_down.weight",
                    "transformer_blocks.{block}.ff.linear_out.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.transformer_blocks.{block}.ff.linear_out.alpha",
                    "transformer.transformer_blocks.{block}.ff.net.2.alpha",
                    "transformer_blocks.{block}.ff.linear_out.alpha",
                ],
            ),
            # Context stream feed-forward
            LoRATarget(
                model_path="transformer_blocks.{block}.ff_context.linear_in",
                possible_up_patterns=[
                    "transformer.transformer_blocks.{block}.ff_context.linear_in.lora_B.weight",
                    "transformer.transformer_blocks.{block}.ff_context.linear_in.lora_B",
                    "transformer.transformer_blocks.{block}.ff_context.net.0.proj.lora_B.weight",
                    "transformer_blocks.{block}.ff_context.linear_in.lora_up.weight",
                    "transformer_blocks.{block}.ff_context.linear_in.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.transformer_blocks.{block}.ff_context.linear_in.lora_A.weight",
                    "transformer.transformer_blocks.{block}.ff_context.linear_in.lora_A",
                    "transformer.transformer_blocks.{block}.ff_context.net.0.proj.lora_A.weight",
                    "transformer_blocks.{block}.ff_context.linear_in.lora_down.weight",
                    "transformer_blocks.{block}.ff_context.linear_in.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.transformer_blocks.{block}.ff_context.linear_in.alpha",
                    "transformer.transformer_blocks.{block}.ff_context.net.0.proj.alpha",
                    "transformer_blocks.{block}.ff_context.linear_in.alpha",
                ],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.ff_context.linear_out",
                possible_up_patterns=[
                    "transformer.transformer_blocks.{block}.ff_context.linear_out.lora_B.weight",
                    "transformer.transformer_blocks.{block}.ff_context.linear_out.lora_B",
                    "transformer.transformer_blocks.{block}.ff_context.net.2.lora_B.weight",
                    "transformer_blocks.{block}.ff_context.linear_out.lora_up.weight",
                    "transformer_blocks.{block}.ff_context.linear_out.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.transformer_blocks.{block}.ff_context.linear_out.lora_A.weight",
                    "transformer.transformer_blocks.{block}.ff_context.linear_out.lora_A",
                    "transformer.transformer_blocks.{block}.ff_context.net.2.lora_A.weight",
                    "transformer_blocks.{block}.ff_context.linear_out.lora_down.weight",
                    "transformer_blocks.{block}.ff_context.linear_out.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.transformer_blocks.{block}.ff_context.linear_out.alpha",
                    "transformer.transformer_blocks.{block}.ff_context.net.2.alpha",
                    "transformer_blocks.{block}.ff_context.linear_out.alpha",
                ],
            ),
        ]

    @staticmethod
    def _get_standard_single_transformer_block_targets() -> list[LoRATarget]:
        """Standard LoRA patterns for single transformer blocks (48 blocks).

        FLUX.2 single blocks use fused to_qkv_mlp_proj and to_out projections.
        """
        return [
            # Fused QKV+MLP projection
            LoRATarget(
                model_path="single_transformer_blocks.{block}.attn.to_qkv_mlp_proj",
                possible_up_patterns=[
                    "transformer.single_transformer_blocks.{block}.attn.to_qkv_mlp_proj.lora_B.weight",
                    "transformer.single_transformer_blocks.{block}.attn.to_qkv_mlp_proj.lora_B",
                    "single_transformer_blocks.{block}.attn.to_qkv_mlp_proj.lora_up.weight",
                    "single_transformer_blocks.{block}.attn.to_qkv_mlp_proj.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.single_transformer_blocks.{block}.attn.to_qkv_mlp_proj.lora_A.weight",
                    "transformer.single_transformer_blocks.{block}.attn.to_qkv_mlp_proj.lora_A",
                    "single_transformer_blocks.{block}.attn.to_qkv_mlp_proj.lora_down.weight",
                    "single_transformer_blocks.{block}.attn.to_qkv_mlp_proj.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.single_transformer_blocks.{block}.attn.to_qkv_mlp_proj.alpha",
                    "single_transformer_blocks.{block}.attn.to_qkv_mlp_proj.alpha",
                ],
            ),
            # Output projection
            LoRATarget(
                model_path="single_transformer_blocks.{block}.attn.to_out",
                possible_up_patterns=[
                    "transformer.single_transformer_blocks.{block}.attn.to_out.lora_B.weight",
                    "transformer.single_transformer_blocks.{block}.attn.to_out.lora_B",
                    "single_transformer_blocks.{block}.attn.to_out.lora_up.weight",
                    "single_transformer_blocks.{block}.attn.to_out.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.single_transformer_blocks.{block}.attn.to_out.lora_A.weight",
                    "transformer.single_transformer_blocks.{block}.attn.to_out.lora_A",
                    "single_transformer_blocks.{block}.attn.to_out.lora_down.weight",
                    "single_transformer_blocks.{block}.attn.to_out.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.single_transformer_blocks.{block}.attn.to_out.alpha",
                    "single_transformer_blocks.{block}.attn.to_out.alpha",
                ],
            ),
        ]

    @staticmethod
    def _get_bfl_transformer_block_targets() -> list[LoRATarget]:
        """BFL-style LoRA patterns for joint transformer blocks (8 blocks)."""
        return [
            # Image attention with fused QKV
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
            # Context/text attention with fused QKV
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
            # Image FFN
            LoRATarget(
                model_path="transformer_blocks.{block}.ff.linear_in",
                possible_up_patterns=["lora_unet_double_blocks_{block}_img_mlp_0.lora_up.weight"],
                possible_down_patterns=["lora_unet_double_blocks_{block}_img_mlp_0.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_double_blocks_{block}_img_mlp_0.alpha"],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.ff.linear_out",
                possible_up_patterns=["lora_unet_double_blocks_{block}_img_mlp_2.lora_up.weight"],
                possible_down_patterns=["lora_unet_double_blocks_{block}_img_mlp_2.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_double_blocks_{block}_img_mlp_2.alpha"],
            ),
            # Context FFN
            LoRATarget(
                model_path="transformer_blocks.{block}.ff_context.linear_in",
                possible_up_patterns=["lora_unet_double_blocks_{block}_txt_mlp_0.lora_up.weight"],
                possible_down_patterns=["lora_unet_double_blocks_{block}_txt_mlp_0.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_double_blocks_{block}_txt_mlp_0.alpha"],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.ff_context.linear_out",
                possible_up_patterns=["lora_unet_double_blocks_{block}_txt_mlp_2.lora_up.weight"],
                possible_down_patterns=["lora_unet_double_blocks_{block}_txt_mlp_2.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_double_blocks_{block}_txt_mlp_2.alpha"],
            ),
        ]

    @staticmethod
    def _get_bfl_single_transformer_block_targets() -> list[LoRATarget]:
        """BFL-style LoRA patterns for single transformer blocks (48 blocks).

        FLUX.2 uses fused to_qkv_mlp_proj, so BFL patterns target linear1/linear2.
        """
        return [
            # Fused QKV+MLP via linear1
            LoRATarget(
                model_path="single_transformer_blocks.{block}.attn.to_qkv_mlp_proj",
                possible_up_patterns=["lora_unet_single_blocks_{block}_linear1.lora_up.weight"],
                possible_down_patterns=["lora_unet_single_blocks_{block}_linear1.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_single_blocks_{block}_linear1.alpha"],
            ),
            # Output projection via linear2
            LoRATarget(
                model_path="single_transformer_blocks.{block}.attn.to_out",
                possible_up_patterns=["lora_unet_single_blocks_{block}_linear2.lora_up.weight"],
                possible_down_patterns=["lora_unet_single_blocks_{block}_linear2.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_single_blocks_{block}_linear2.alpha"],
            ),
        ]

    @staticmethod
    def _get_global_modulation_targets() -> list[LoRATarget]:
        """LoRA targets for global modulation layers (FLUX.2 specific).

        FLUX.2 has single global modulation layers (not per-block).
        These are powerful LoRA targets as they affect all blocks.
        """
        return [
            # Double stream image modulation
            LoRATarget(
                model_path="double_stream_modulation_img.linear",
                possible_up_patterns=[
                    "transformer.double_stream_modulation_img.linear.lora_B.weight",
                    "transformer.double_stream_modulation_img.linear.lora_B",
                    "double_stream_modulation_img.linear.lora_up.weight",
                    "double_stream_modulation_img.linear.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.double_stream_modulation_img.linear.lora_A.weight",
                    "transformer.double_stream_modulation_img.linear.lora_A",
                    "double_stream_modulation_img.linear.lora_down.weight",
                    "double_stream_modulation_img.linear.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.double_stream_modulation_img.linear.alpha",
                    "double_stream_modulation_img.linear.alpha",
                ],
            ),
            # Double stream text modulation
            LoRATarget(
                model_path="double_stream_modulation_txt.linear",
                possible_up_patterns=[
                    "transformer.double_stream_modulation_txt.linear.lora_B.weight",
                    "transformer.double_stream_modulation_txt.linear.lora_B",
                    "double_stream_modulation_txt.linear.lora_up.weight",
                    "double_stream_modulation_txt.linear.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.double_stream_modulation_txt.linear.lora_A.weight",
                    "transformer.double_stream_modulation_txt.linear.lora_A",
                    "double_stream_modulation_txt.linear.lora_down.weight",
                    "double_stream_modulation_txt.linear.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.double_stream_modulation_txt.linear.alpha",
                    "double_stream_modulation_txt.linear.alpha",
                ],
            ),
            # Single stream modulation
            LoRATarget(
                model_path="single_stream_modulation.linear",
                possible_up_patterns=[
                    "transformer.single_stream_modulation.linear.lora_B.weight",
                    "transformer.single_stream_modulation.linear.lora_B",
                    "single_stream_modulation.linear.lora_up.weight",
                    "single_stream_modulation.linear.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.single_stream_modulation.linear.lora_A.weight",
                    "transformer.single_stream_modulation.linear.lora_A",
                    "single_stream_modulation.linear.lora_down.weight",
                    "single_stream_modulation.linear.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.single_stream_modulation.linear.alpha",
                    "single_stream_modulation.linear.alpha",
                ],
            ),
        ]
