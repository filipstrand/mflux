from mlx import nn

from mflux.models.qwen.model.qwen_transformer.qwen_transformer_block import QwenTransformerBlock


class QwenTransformerBlockApplier:
    @staticmethod
    def apply_from_handler(module: QwenTransformerBlock, block_weights: dict) -> None:
        # img_mod.1.* and txt_mod.1.*
        if "img_mod" in block_weights and "1" in block_weights["img_mod"]:
            lin = block_weights["img_mod"]["1"]
            if "weight" in lin:
                module.img_mod_linear.weight = lin["weight"]
            if "bias" in lin:
                module.img_mod_linear.bias = lin["bias"]
        if "txt_mod" in block_weights and "1" in block_weights["txt_mod"]:
            lin = block_weights["txt_mod"]["1"]
            if "weight" in lin:
                module.txt_mod_linear.weight = lin["weight"]
            if "bias" in lin:
                module.txt_mod_linear.bias = lin["bias"]

        attn = block_weights.get("attn", {})

        def set_lin(target: nn.Linear, src: dict):
            if "weight" in src:
                # PT and MLX both store [out,in] weights for Linear in our usage
                target.weight = src["weight"]
            if "bias" in src:
                target.bias = src["bias"]

        if "to_q" in attn:
            set_lin(module.to_q, attn["to_q"])
        if "to_k" in attn:
            set_lin(module.to_k, attn["to_k"])
        if "to_v" in attn:
            set_lin(module.to_v, attn["to_v"])
        if "add_q_proj" in attn:
            set_lin(module.add_q_proj, attn["add_q_proj"])
        if "add_k_proj" in attn:
            set_lin(module.add_k_proj, attn["add_k_proj"])
        if "add_v_proj" in attn:
            set_lin(module.add_v_proj, attn["add_v_proj"])

        if "norm_q" in attn and "weight" in attn["norm_q"]:
            module.norm_q.weight = attn["norm_q"]["weight"]
        if "norm_k" in attn and "weight" in attn["norm_k"]:
            module.norm_k.weight = attn["norm_k"]["weight"]
        if "norm_added_q" in attn and "weight" in attn["norm_added_q"]:
            module.norm_added_q.weight = attn["norm_added_q"]["weight"]
        if "norm_added_k" in attn and "weight" in attn["norm_added_k"]:
            module.norm_added_k.weight = attn["norm_added_k"]["weight"]

        if "to_out" in attn and len(attn["to_out"]) > 0:
            to_out0 = attn["to_out"][0]
            set_lin(module.attn_to_out[0], to_out0)
        if "to_add_out" in attn:
            set_lin(module.to_add_out, attn["to_add_out"])

        # MLPs: net.0.proj (in), net.2 (out)
        img_mlp = block_weights.get("img_mlp", {}).get("net", {})
        if "0" in img_mlp and "proj" in img_mlp["0"]:
            set_lin(module.img_mlp_in, img_mlp["0"]["proj"])
        if "2" in img_mlp:
            set_lin(module.img_mlp_out, img_mlp["2"])

        txt_mlp = block_weights.get("txt_mlp", {}).get("net", {})
        if "0" in txt_mlp and "proj" in txt_mlp["0"]:
            set_lin(module.txt_mlp_in, txt_mlp["0"]["proj"])
        if "2" in txt_mlp:
            set_lin(module.txt_mlp_out, txt_mlp["2"])