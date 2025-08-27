import mlx.core as mx
from mlx import nn

from mflux.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer
from mflux.models.qwen.model.qwen_transformer.qwen_transformer_block_applier import QwenTransformerBlockApplier


class QwenImageTransformerApplier:
    @staticmethod
    def apply_from_handler(module: QwenTransformer, weights: dict) -> None:
        def set_linear_pt_to_mlx(target: nn.Linear, w: mx.array | None, b: mx.array | None) -> None:
            # PT saves Linear weight as [out, in]; MLX stores weight as [out, in] as well
            if w is not None:
                target.weight = w
            if b is not None:
                target.bias = b

        # Top-level
        if "img_in" in weights:
            w = weights["img_in"].get("weight")
            b = weights["img_in"].get("bias")
            set_linear_pt_to_mlx(module.img_in, w, b)
        if "txt_norm" in weights and "weight" in weights["txt_norm"]:
            module.txt_norm.weight = weights["txt_norm"]["weight"]
        if "txt_in" in weights:
            w = weights["txt_in"].get("weight")
            b = weights["txt_in"].get("bias")
            set_linear_pt_to_mlx(module.txt_in, w, b)

        # Time/text embedder
        tte = weights.get("time_text_embed", {}).get("timestep_embedder", {})
        l1 = tte.get("linear_1", {})
        l2 = tte.get("linear_2", {})
        if "weight" in l1 or "bias" in l1:
            set_linear_pt_to_mlx(module.time_text_embed.timestep_embedder.linear_1, l1.get("weight"), l1.get("bias"))
        if "weight" in l2 or "bias" in l2:
            set_linear_pt_to_mlx(module.time_text_embed.timestep_embedder.linear_2, l2.get("weight"), l2.get("bias"))

        # Blocks
        blocks = weights.get("transformer_blocks", [])
        for i, bw in enumerate(blocks):
            if i >= len(module.transformer_blocks):
                break
            QwenTransformerBlockApplier.apply_from_handler(module.transformer_blocks[i], bw)

        # Output head
        outw = weights.get("output", {})
        norm_out = outw.get("norm_out", {})
        # AdaLayerNormContinuous.linear
        # AdaLayerNormContinuous contains a Linear; PT and MLX share [out,in]
        if "linear.weight" in norm_out:
            module.norm_out.linear.weight = norm_out["linear.weight"]
        if "linear.bias" in norm_out:
            module.norm_out.linear.bias = norm_out["linear.bias"]
        if "proj_out" in outw:
            po = outw["proj_out"]
            set_linear_pt_to_mlx(module.proj_out, po.get("weight"), po.get("bias"))
