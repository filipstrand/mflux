"""LoRA mapping for Hunyuan-DiT model.

Defines LoRA targets for fine-tuning Hunyuan-DiT.

LoRA targets include:
- Self-attention (attn1): to_q, to_k, to_v, to_out
- Cross-attention (attn2): to_q, to_k, to_v, to_out
- Feed-forward: net_0_proj, net_2
"""

from typing import List

from mflux.models.common.lora.mapping.lora_mapping import LoRATarget


class HunyuanLoRAMapping:
    """LoRA mapping for Hunyuan-DiT model."""

    NUM_BLOCKS = 28

    @staticmethod
    def get_mapping() -> List[LoRATarget]:
        """
        Get LoRA targets for Hunyuan-DiT.

        Standard diffusers format patterns:
        - lora_unet_down_blocks_{block}_attentions_{attn}_transformer_blocks_{tb}_{layer}
        - For Hunyuan specifically: blocks.{block}.attn{1,2}.{layer}

        Returns:
            List of LoRA targets covering all attention and FFN layers
        """
        targets = []

        # DiT blocks (28 blocks)
        for block_idx in range(HunyuanLoRAMapping.NUM_BLOCKS):
            # Self-attention (attn1)
            targets.extend([
                LoRATarget(
                    model_path=f"blocks.{block_idx}.attn1.to_q",
                    possible_up_patterns=[
                        f"lora_unet_blocks_{block_idx}_attn1_to_q.lora_up.weight",
                        f"blocks.{block_idx}.attn1.to_q.lora_up.weight",
                    ],
                    possible_down_patterns=[
                        f"lora_unet_blocks_{block_idx}_attn1_to_q.lora_down.weight",
                        f"blocks.{block_idx}.attn1.to_q.lora_down.weight",
                    ],
                    possible_alpha_patterns=[
                        f"lora_unet_blocks_{block_idx}_attn1_to_q.alpha",
                        f"blocks.{block_idx}.attn1.to_q.alpha",
                    ],
                ),
                LoRATarget(
                    model_path=f"blocks.{block_idx}.attn1.to_k",
                    possible_up_patterns=[
                        f"lora_unet_blocks_{block_idx}_attn1_to_k.lora_up.weight",
                        f"blocks.{block_idx}.attn1.to_k.lora_up.weight",
                    ],
                    possible_down_patterns=[
                        f"lora_unet_blocks_{block_idx}_attn1_to_k.lora_down.weight",
                        f"blocks.{block_idx}.attn1.to_k.lora_down.weight",
                    ],
                    possible_alpha_patterns=[
                        f"lora_unet_blocks_{block_idx}_attn1_to_k.alpha",
                        f"blocks.{block_idx}.attn1.to_k.alpha",
                    ],
                ),
                LoRATarget(
                    model_path=f"blocks.{block_idx}.attn1.to_v",
                    possible_up_patterns=[
                        f"lora_unet_blocks_{block_idx}_attn1_to_v.lora_up.weight",
                        f"blocks.{block_idx}.attn1.to_v.lora_up.weight",
                    ],
                    possible_down_patterns=[
                        f"lora_unet_blocks_{block_idx}_attn1_to_v.lora_down.weight",
                        f"blocks.{block_idx}.attn1.to_v.lora_down.weight",
                    ],
                    possible_alpha_patterns=[
                        f"lora_unet_blocks_{block_idx}_attn1_to_v.alpha",
                        f"blocks.{block_idx}.attn1.to_v.alpha",
                    ],
                ),
                LoRATarget(
                    model_path=f"blocks.{block_idx}.attn1.to_out",
                    possible_up_patterns=[
                        f"lora_unet_blocks_{block_idx}_attn1_to_out_0.lora_up.weight",
                        f"blocks.{block_idx}.attn1.to_out.lora_up.weight",
                    ],
                    possible_down_patterns=[
                        f"lora_unet_blocks_{block_idx}_attn1_to_out_0.lora_down.weight",
                        f"blocks.{block_idx}.attn1.to_out.lora_down.weight",
                    ],
                    possible_alpha_patterns=[
                        f"lora_unet_blocks_{block_idx}_attn1_to_out_0.alpha",
                        f"blocks.{block_idx}.attn1.to_out.alpha",
                    ],
                ),
            ])

            # Cross-attention (attn2)
            targets.extend([
                LoRATarget(
                    model_path=f"blocks.{block_idx}.attn2.to_q",
                    possible_up_patterns=[
                        f"lora_unet_blocks_{block_idx}_attn2_to_q.lora_up.weight",
                        f"blocks.{block_idx}.attn2.to_q.lora_up.weight",
                    ],
                    possible_down_patterns=[
                        f"lora_unet_blocks_{block_idx}_attn2_to_q.lora_down.weight",
                        f"blocks.{block_idx}.attn2.to_q.lora_down.weight",
                    ],
                    possible_alpha_patterns=[
                        f"lora_unet_blocks_{block_idx}_attn2_to_q.alpha",
                        f"blocks.{block_idx}.attn2.to_q.alpha",
                    ],
                ),
                LoRATarget(
                    model_path=f"blocks.{block_idx}.attn2.to_k",
                    possible_up_patterns=[
                        f"lora_unet_blocks_{block_idx}_attn2_to_k.lora_up.weight",
                        f"blocks.{block_idx}.attn2.to_k.lora_up.weight",
                    ],
                    possible_down_patterns=[
                        f"lora_unet_blocks_{block_idx}_attn2_to_k.lora_down.weight",
                        f"blocks.{block_idx}.attn2.to_k.lora_down.weight",
                    ],
                    possible_alpha_patterns=[
                        f"lora_unet_blocks_{block_idx}_attn2_to_k.alpha",
                        f"blocks.{block_idx}.attn2.to_k.alpha",
                    ],
                ),
                LoRATarget(
                    model_path=f"blocks.{block_idx}.attn2.to_v",
                    possible_up_patterns=[
                        f"lora_unet_blocks_{block_idx}_attn2_to_v.lora_up.weight",
                        f"blocks.{block_idx}.attn2.to_v.lora_up.weight",
                    ],
                    possible_down_patterns=[
                        f"lora_unet_blocks_{block_idx}_attn2_to_v.lora_down.weight",
                        f"blocks.{block_idx}.attn2.to_v.lora_down.weight",
                    ],
                    possible_alpha_patterns=[
                        f"lora_unet_blocks_{block_idx}_attn2_to_v.alpha",
                        f"blocks.{block_idx}.attn2.to_v.alpha",
                    ],
                ),
                LoRATarget(
                    model_path=f"blocks.{block_idx}.attn2.to_out",
                    possible_up_patterns=[
                        f"lora_unet_blocks_{block_idx}_attn2_to_out_0.lora_up.weight",
                        f"blocks.{block_idx}.attn2.to_out.lora_up.weight",
                    ],
                    possible_down_patterns=[
                        f"lora_unet_blocks_{block_idx}_attn2_to_out_0.lora_down.weight",
                        f"blocks.{block_idx}.attn2.to_out.lora_down.weight",
                    ],
                    possible_alpha_patterns=[
                        f"lora_unet_blocks_{block_idx}_attn2_to_out_0.alpha",
                        f"blocks.{block_idx}.attn2.to_out.alpha",
                    ],
                ),
            ])

            # Feed-forward network
            targets.extend([
                LoRATarget(
                    model_path=f"blocks.{block_idx}.ff.net_0_proj",
                    possible_up_patterns=[
                        f"lora_unet_blocks_{block_idx}_ff_net_0_proj.lora_up.weight",
                        f"blocks.{block_idx}.ff.net.0.proj.lora_up.weight",
                    ],
                    possible_down_patterns=[
                        f"lora_unet_blocks_{block_idx}_ff_net_0_proj.lora_down.weight",
                        f"blocks.{block_idx}.ff.net.0.proj.lora_down.weight",
                    ],
                    possible_alpha_patterns=[
                        f"lora_unet_blocks_{block_idx}_ff_net_0_proj.alpha",
                        f"blocks.{block_idx}.ff.net.0.proj.alpha",
                    ],
                ),
                LoRATarget(
                    model_path=f"blocks.{block_idx}.ff.net_2",
                    possible_up_patterns=[
                        f"lora_unet_blocks_{block_idx}_ff_net_2.lora_up.weight",
                        f"blocks.{block_idx}.ff.net.2.lora_up.weight",
                    ],
                    possible_down_patterns=[
                        f"lora_unet_blocks_{block_idx}_ff_net_2.lora_down.weight",
                        f"blocks.{block_idx}.ff.net.2.lora_down.weight",
                    ],
                    possible_alpha_patterns=[
                        f"lora_unet_blocks_{block_idx}_ff_net_2.alpha",
                        f"blocks.{block_idx}.ff.net.2.alpha",
                    ],
                ),
            ])

        return targets
