"""LoRA mapping for NewBie-image model.

Defines LoRA targets for fine-tuning NextDiT.

LoRA targets include:
- Self-attention (attn1): wq, wk, wv, wo (GQA)
- Cross-attention (attn2): wq, wk, wv, wo (GQA)
- Feed-forward: w1, w2, w3 (SwiGLU)
"""

from typing import List

from mflux.models.common.lora.mapping.lora_mapping import LoRATarget


class NewBieLoRAMapping:
    """LoRA mapping for NewBie-image NextDiT model."""

    NUM_BLOCKS = 36

    @staticmethod
    def get_mapping() -> List[LoRATarget]:
        """
        Get LoRA targets for NewBie-image NextDiT.

        Standard patterns:
        - Lumina format: layers.{block}.attention.{layer}
        - Diffusers format: transformer_blocks.{block}.attn{1,2}.{layer}

        Returns:
            List of LoRA targets covering all attention and FFN layers
        """
        targets = []

        # NextDiT blocks (36 blocks)
        for block_idx in range(NewBieLoRAMapping.NUM_BLOCKS):
            # Self-attention (attn1) - GQA
            targets.extend([
                LoRATarget(
                    model_path=f"blocks.{block_idx}.attn1.wq",
                    possible_up_patterns=[
                        f"lora_unet_layers_{block_idx}_attention_wq.lora_up.weight",
                        f"layers.{block_idx}.attention.wq.lora_up.weight",
                        f"transformer_blocks.{block_idx}.attn1.to_q.lora_up.weight",
                    ],
                    possible_down_patterns=[
                        f"lora_unet_layers_{block_idx}_attention_wq.lora_down.weight",
                        f"layers.{block_idx}.attention.wq.lora_down.weight",
                        f"transformer_blocks.{block_idx}.attn1.to_q.lora_down.weight",
                    ],
                    possible_alpha_patterns=[
                        f"lora_unet_layers_{block_idx}_attention_wq.alpha",
                        f"layers.{block_idx}.attention.wq.alpha",
                        f"transformer_blocks.{block_idx}.attn1.to_q.alpha",
                    ],
                ),
                LoRATarget(
                    model_path=f"blocks.{block_idx}.attn1.wk",
                    possible_up_patterns=[
                        f"lora_unet_layers_{block_idx}_attention_wk.lora_up.weight",
                        f"layers.{block_idx}.attention.wk.lora_up.weight",
                        f"transformer_blocks.{block_idx}.attn1.to_k.lora_up.weight",
                    ],
                    possible_down_patterns=[
                        f"lora_unet_layers_{block_idx}_attention_wk.lora_down.weight",
                        f"layers.{block_idx}.attention.wk.lora_down.weight",
                        f"transformer_blocks.{block_idx}.attn1.to_k.lora_down.weight",
                    ],
                    possible_alpha_patterns=[
                        f"lora_unet_layers_{block_idx}_attention_wk.alpha",
                        f"layers.{block_idx}.attention.wk.alpha",
                        f"transformer_blocks.{block_idx}.attn1.to_k.alpha",
                    ],
                ),
                LoRATarget(
                    model_path=f"blocks.{block_idx}.attn1.wv",
                    possible_up_patterns=[
                        f"lora_unet_layers_{block_idx}_attention_wv.lora_up.weight",
                        f"layers.{block_idx}.attention.wv.lora_up.weight",
                        f"transformer_blocks.{block_idx}.attn1.to_v.lora_up.weight",
                    ],
                    possible_down_patterns=[
                        f"lora_unet_layers_{block_idx}_attention_wv.lora_down.weight",
                        f"layers.{block_idx}.attention.wv.lora_down.weight",
                        f"transformer_blocks.{block_idx}.attn1.to_v.lora_down.weight",
                    ],
                    possible_alpha_patterns=[
                        f"lora_unet_layers_{block_idx}_attention_wv.alpha",
                        f"layers.{block_idx}.attention.wv.alpha",
                        f"transformer_blocks.{block_idx}.attn1.to_v.alpha",
                    ],
                ),
                LoRATarget(
                    model_path=f"blocks.{block_idx}.attn1.wo",
                    possible_up_patterns=[
                        f"lora_unet_layers_{block_idx}_attention_wo.lora_up.weight",
                        f"layers.{block_idx}.attention.wo.lora_up.weight",
                        f"transformer_blocks.{block_idx}.attn1.to_out_0.lora_up.weight",
                    ],
                    possible_down_patterns=[
                        f"lora_unet_layers_{block_idx}_attention_wo.lora_down.weight",
                        f"layers.{block_idx}.attention.wo.lora_down.weight",
                        f"transformer_blocks.{block_idx}.attn1.to_out_0.lora_down.weight",
                    ],
                    possible_alpha_patterns=[
                        f"lora_unet_layers_{block_idx}_attention_wo.alpha",
                        f"layers.{block_idx}.attention.wo.alpha",
                        f"transformer_blocks.{block_idx}.attn1.to_out_0.alpha",
                    ],
                ),
            ])

            # Cross-attention (attn2) - GQA
            targets.extend([
                LoRATarget(
                    model_path=f"blocks.{block_idx}.attn2.wq",
                    possible_up_patterns=[
                        f"lora_unet_layers_{block_idx}_cross_attention_wq.lora_up.weight",
                        f"layers.{block_idx}.cross_attention.wq.lora_up.weight",
                        f"transformer_blocks.{block_idx}.attn2.to_q.lora_up.weight",
                    ],
                    possible_down_patterns=[
                        f"lora_unet_layers_{block_idx}_cross_attention_wq.lora_down.weight",
                        f"layers.{block_idx}.cross_attention.wq.lora_down.weight",
                        f"transformer_blocks.{block_idx}.attn2.to_q.lora_down.weight",
                    ],
                    possible_alpha_patterns=[
                        f"lora_unet_layers_{block_idx}_cross_attention_wq.alpha",
                        f"layers.{block_idx}.cross_attention.wq.alpha",
                        f"transformer_blocks.{block_idx}.attn2.to_q.alpha",
                    ],
                ),
                LoRATarget(
                    model_path=f"blocks.{block_idx}.attn2.wk",
                    possible_up_patterns=[
                        f"lora_unet_layers_{block_idx}_cross_attention_wk.lora_up.weight",
                        f"layers.{block_idx}.cross_attention.wk.lora_up.weight",
                        f"transformer_blocks.{block_idx}.attn2.to_k.lora_up.weight",
                    ],
                    possible_down_patterns=[
                        f"lora_unet_layers_{block_idx}_cross_attention_wk.lora_down.weight",
                        f"layers.{block_idx}.cross_attention.wk.lora_down.weight",
                        f"transformer_blocks.{block_idx}.attn2.to_k.lora_down.weight",
                    ],
                    possible_alpha_patterns=[
                        f"lora_unet_layers_{block_idx}_cross_attention_wk.alpha",
                        f"layers.{block_idx}.cross_attention.wk.alpha",
                        f"transformer_blocks.{block_idx}.attn2.to_k.alpha",
                    ],
                ),
                LoRATarget(
                    model_path=f"blocks.{block_idx}.attn2.wv",
                    possible_up_patterns=[
                        f"lora_unet_layers_{block_idx}_cross_attention_wv.lora_up.weight",
                        f"layers.{block_idx}.cross_attention.wv.lora_up.weight",
                        f"transformer_blocks.{block_idx}.attn2.to_v.lora_up.weight",
                    ],
                    possible_down_patterns=[
                        f"lora_unet_layers_{block_idx}_cross_attention_wv.lora_down.weight",
                        f"layers.{block_idx}.cross_attention.wv.lora_down.weight",
                        f"transformer_blocks.{block_idx}.attn2.to_v.lora_down.weight",
                    ],
                    possible_alpha_patterns=[
                        f"lora_unet_layers_{block_idx}_cross_attention_wv.alpha",
                        f"layers.{block_idx}.cross_attention.wv.alpha",
                        f"transformer_blocks.{block_idx}.attn2.to_v.alpha",
                    ],
                ),
                LoRATarget(
                    model_path=f"blocks.{block_idx}.attn2.wo",
                    possible_up_patterns=[
                        f"lora_unet_layers_{block_idx}_cross_attention_wo.lora_up.weight",
                        f"layers.{block_idx}.cross_attention.wo.lora_up.weight",
                        f"transformer_blocks.{block_idx}.attn2.to_out_0.lora_up.weight",
                    ],
                    possible_down_patterns=[
                        f"lora_unet_layers_{block_idx}_cross_attention_wo.lora_down.weight",
                        f"layers.{block_idx}.cross_attention.wo.lora_down.weight",
                        f"transformer_blocks.{block_idx}.attn2.to_out_0.lora_down.weight",
                    ],
                    possible_alpha_patterns=[
                        f"lora_unet_layers_{block_idx}_cross_attention_wo.alpha",
                        f"layers.{block_idx}.cross_attention.wo.alpha",
                        f"transformer_blocks.{block_idx}.attn2.to_out_0.alpha",
                    ],
                ),
            ])

            # SwiGLU Feed-forward network
            targets.extend([
                LoRATarget(
                    model_path=f"blocks.{block_idx}.ffn.w1",
                    possible_up_patterns=[
                        f"lora_unet_layers_{block_idx}_feed_forward_w1.lora_up.weight",
                        f"layers.{block_idx}.feed_forward.w1.lora_up.weight",
                        f"transformer_blocks.{block_idx}.ff_net_0_proj.lora_up.weight",
                    ],
                    possible_down_patterns=[
                        f"lora_unet_layers_{block_idx}_feed_forward_w1.lora_down.weight",
                        f"layers.{block_idx}.feed_forward.w1.lora_down.weight",
                        f"transformer_blocks.{block_idx}.ff_net_0_proj.lora_down.weight",
                    ],
                    possible_alpha_patterns=[
                        f"lora_unet_layers_{block_idx}_feed_forward_w1.alpha",
                        f"layers.{block_idx}.feed_forward.w1.alpha",
                        f"transformer_blocks.{block_idx}.ff_net_0_proj.alpha",
                    ],
                ),
                LoRATarget(
                    model_path=f"blocks.{block_idx}.ffn.w2",
                    possible_up_patterns=[
                        f"lora_unet_layers_{block_idx}_feed_forward_w2.lora_up.weight",
                        f"layers.{block_idx}.feed_forward.w2.lora_up.weight",
                        f"transformer_blocks.{block_idx}.ff_net_2.lora_up.weight",
                    ],
                    possible_down_patterns=[
                        f"lora_unet_layers_{block_idx}_feed_forward_w2.lora_down.weight",
                        f"layers.{block_idx}.feed_forward.w2.lora_down.weight",
                        f"transformer_blocks.{block_idx}.ff_net_2.lora_down.weight",
                    ],
                    possible_alpha_patterns=[
                        f"lora_unet_layers_{block_idx}_feed_forward_w2.alpha",
                        f"layers.{block_idx}.feed_forward.w2.alpha",
                        f"transformer_blocks.{block_idx}.ff_net_2.alpha",
                    ],
                ),
                LoRATarget(
                    model_path=f"blocks.{block_idx}.ffn.w3",
                    possible_up_patterns=[
                        f"lora_unet_layers_{block_idx}_feed_forward_w3.lora_up.weight",
                        f"layers.{block_idx}.feed_forward.w3.lora_up.weight",
                        f"transformer_blocks.{block_idx}.ff_net_0_gate.lora_up.weight",
                    ],
                    possible_down_patterns=[
                        f"lora_unet_layers_{block_idx}_feed_forward_w3.lora_down.weight",
                        f"layers.{block_idx}.feed_forward.w3.lora_down.weight",
                        f"transformer_blocks.{block_idx}.ff_net_0_gate.lora_down.weight",
                    ],
                    possible_alpha_patterns=[
                        f"lora_unet_layers_{block_idx}_feed_forward_w3.alpha",
                        f"layers.{block_idx}.feed_forward.w3.alpha",
                        f"transformer_blocks.{block_idx}.ff_net_0_gate.alpha",
                    ],
                ),
            ])

        return targets
