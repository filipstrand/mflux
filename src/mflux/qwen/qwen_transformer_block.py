from __future__ import annotations

import mlx.core as mx
from mlx import nn

from mflux.models.transformer.common.attention_utils import AttentionUtils


class QwenTransformerBlockMLX(nn.Module):
    def __init__(self, dim: int = 3072, num_heads: int = 24, head_dim: int = 128):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.debug: bool = False

        # Modulation (SiLU + Linear to 6*dim for shift/scale/gate twice)
        self.img_mod_linear = nn.Linear(dim, 6 * dim)
        self.txt_mod_linear = nn.Linear(dim, 6 * dim)

        # Norms (elementwise_affine False)
        self.img_norm1 = nn.LayerNorm(dims=dim, eps=1e-6, affine=False)
        self.txt_norm1 = nn.LayerNorm(dims=dim, eps=1e-6, affine=False)

        # Attention projections and norms
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.add_q_proj = nn.Linear(dim, dim)
        self.add_k_proj = nn.Linear(dim, dim)
        self.add_v_proj = nn.Linear(dim, dim)

        self.norm_q = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.norm_k = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.norm_added_q = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.norm_added_k = nn.RMSNorm(self.head_dim, eps=1e-6)

        self.attn_to_out = [nn.Linear(dim, dim)]
        self.to_add_out = nn.Linear(dim, dim)

        # Second-stage norms and MLPs
        self.img_norm2 = nn.LayerNorm(dims=dim, eps=1e-6, affine=False)
        self.txt_norm2 = nn.LayerNorm(dims=dim, eps=1e-6, affine=False)

        self.img_mlp_in = nn.Linear(dim, 4 * dim)
        self.img_mlp_out = nn.Linear(4 * dim, dim)

        self.txt_mlp_in = nn.Linear(dim, 4 * dim)
        self.txt_mlp_out = nn.Linear(4 * dim, dim)

    @staticmethod
    def _modulate(x: mx.array, mod_params: mx.array) -> tuple[mx.array, mx.array]:
        # mod_params: [B, 3*dim] -> shift, scale, gate each [B, dim]
        shift, scale, gate = mx.split(mod_params, 3, axis=-1)
        return x * (1 + scale[:, None, :]) + shift[:, None, :], gate

    def __call__(
        self,
        hidden_states: mx.array,  # [B, S_img, dim]
        encoder_hidden_states: mx.array,  # [B, S_txt, dim]
        encoder_hidden_states_mask: mx.array | None,
        temb: mx.array,  # [B, dim]
        image_rotary_emb: tuple[mx.array, mx.array],  # (img_rot: [1,1,S_img,D/2,2,2], txt_rot: [1,1,S_txt,D/2,2,2])
    ) -> tuple[mx.array, mx.array]:
        # Modulation params
        img_mod_params = self.img_mod_linear(nn.silu(temb))  # [B, 6*dim]
        txt_mod_params = self.txt_mod_linear(nn.silu(temb))  # [B, 6*dim]

        img_mod1, img_mod2 = mx.split(img_mod_params, 2, axis=-1)  # each [B, 3*dim]
        txt_mod1, txt_mod2 = mx.split(txt_mod_params, 2, axis=-1)

        # Norm + modulate (stage 1)
        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1)

        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

        # Attention: compute QKV for image and text streams
        img_q, img_k, img_v = AttentionUtils.process_qkv(
            hidden_states=img_modulated,
            to_q=self.to_q,
            to_k=self.to_k,
            to_v=self.to_v,
            norm_q=self.norm_q,
            norm_k=self.norm_k,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )

        txt_q, txt_k, txt_v = AttentionUtils.process_qkv(
            hidden_states=txt_modulated,
            to_q=self.add_q_proj,
            to_k=self.add_k_proj,
            to_v=self.add_v_proj,
            norm_q=self.norm_added_q,
            norm_k=self.norm_added_k,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )

        if self.debug:
            try:
                iq = img_q[0, 0, 0, :8].astype(mx.float32)
                ik = img_k[0, 0, 0, :8].astype(mx.float32)
                tq = txt_q[0, 0, 0, :8].astype(mx.float32)
                tk = txt_k[0, 0, 0, :8].astype(mx.float32)
                # print(f"🔎 MLX attn pre-rope img_q[:8]={np.array(iq)}, img_k[:8]={np.array(ik)}")
                # print(f"🔎 MLX attn pre-rope txt_q[:8]={np.array(tq)}, txt_k[:8]={np.array(tk)}")
            except Exception:
                pass

        # Apply RoPE separately in [B,H,S,D] layout using real mixing with per-stream lengths
        img_rot, txt_rot = image_rotary_emb  # rot mats [1,1,S,D/2,2,2]
        # Convert rot mats to cos/sin [S,D/2]
        img_cos = img_rot[..., 0, 0].reshape(img_rot.shape[2], img_rot.shape[3])
        img_sin = img_rot[..., 1, 0].reshape(img_rot.shape[2], img_rot.shape[3])
        txt_cos = txt_rot[..., 0, 0].reshape(txt_rot.shape[2], txt_rot.shape[3])
        txt_sin = txt_rot[..., 1, 0].reshape(txt_rot.shape[2], txt_rot.shape[3])

        # Transpose [B,H,S,D] -> [B,S,H,D] for mixing convenience
        img_q_bshd = mx.transpose(img_q, (0, 2, 1, 3))
        img_k_bshd = mx.transpose(img_k, (0, 2, 1, 3))
        txt_q_bshd = mx.transpose(txt_q, (0, 2, 1, 3))
        txt_k_bshd = mx.transpose(txt_k, (0, 2, 1, 3))

        img_q_bshd, img_k_bshd = AttentionUtils.apply_rope_bshd(img_q_bshd, img_k_bshd, img_cos, img_sin)
        txt_q_bshd, txt_k_bshd = AttentionUtils.apply_rope_bshd(txt_q_bshd, txt_k_bshd, txt_cos, txt_sin)

        # Back to [B,H,S,D]
        img_q = mx.transpose(img_q_bshd, (0, 2, 1, 3))
        img_k = mx.transpose(img_k_bshd, (0, 2, 1, 3))
        txt_q = mx.transpose(txt_q_bshd, (0, 2, 1, 3))
        txt_k = mx.transpose(txt_k_bshd, (0, 2, 1, 3))

        if self.debug:
            try:
                iq = img_q[0, 0, 0, :8].astype(mx.float32)
                ik = img_k[0, 0, 0, :8].astype(mx.float32)
                tq = txt_q[0, 0, 0, :8].astype(mx.float32)
                tk = txt_k[0, 0, 0, :8].astype(mx.float32)
                # print(f"🔎 MLX attn post-rope img_q[:8]={np.array(iq)}, img_k[:8]={np.array(ik)}")
                # print(f"🔎 MLX attn post-rope txt_q[:8]={np.array(tq)}, txt_k[:8]={np.array(tk)}")
            except Exception:
                pass

        # Concatenate [text, image]
        joint_q = mx.concatenate([txt_q, img_q], axis=2)
        joint_k = mx.concatenate([txt_k, img_k], axis=2)
        joint_v = mx.concatenate([txt_v, img_v], axis=2)

        key_padding_mask = None
        if encoder_hidden_states_mask is not None:
            # Mask applies to text positions only; shape should be [B, S_txt]
            # Build joint mask of shape [B, S_txt + S_img], ones for image positions
            bsz = hidden_states.shape[0]
            s_img = hidden_states.shape[1]
            s_txt = encoder_hidden_states.shape[1]
            ones_img = mx.ones((bsz, s_img), dtype=mx.float32)
            key_padding_mask = mx.concatenate([encoder_hidden_states_mask.astype(mx.float32), ones_img], axis=1)

        # Compute attention explicitly (optionally with key padding mask)
        joint_hs = AttentionUtils.compute_attention_explicit(joint_q, joint_k, joint_v, key_padding_mask)

        # CRITICAL: Convert back to query dtype like reference implementation
        # Reference line 375: joint_hidden_states = joint_hidden_states.to(joint_query.dtype)
        joint_hs = joint_hs.astype(joint_q.dtype)

        if self.debug:
            try:
                j0 = joint_hs[0, 0, :8].astype(mx.float32)
                j1 = joint_hs[0, -1, :8].astype(mx.float32)
                # print(f"🔎 MLX attn joint_hs[0][:8]={np.array(j0)} ... joint_hs[-1][:8]={np.array(j1)}")
            except Exception:
                pass

        # Split back
        seq_txt = encoder_hidden_states.shape[1]
        txt_attn_output = joint_hs[:, :seq_txt, :]
        img_attn_output = joint_hs[:, seq_txt:, :]

        # Output projections (compute in float32 like working version, then cast back)
        img_dtype = img_attn_output.dtype
        txt_dtype = txt_attn_output.dtype
        img_attn_output = self.attn_to_out[0](img_attn_output.astype(mx.float32)).astype(img_dtype)
        txt_attn_output = self.to_add_out(txt_attn_output.astype(mx.float32)).astype(txt_dtype)

        # Apply gates and residual (stage 1)
        hidden_states = hidden_states + img_gate1[:, None, :] * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1[:, None, :] * txt_attn_output

        # Stage 2: norm + MLP for both streams
        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2)
        # MLP compute in float32 like working version
        img_mlp_hidden = self.img_mlp_in(img_modulated2.astype(mx.float32))
        img_mlp_hidden = nn.gelu_approx(img_mlp_hidden)
        img_mlp_out = self.img_mlp_out(img_mlp_hidden).astype(hidden_states.dtype)
        hidden_states = hidden_states + img_gate2[:, None, :] * img_mlp_out

        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
        # MLP compute in float32 like working version
        txt_mlp_hidden = self.txt_mlp_in(txt_modulated2.astype(mx.float32))
        txt_mlp_hidden = nn.gelu_approx(txt_mlp_hidden)
        txt_mlp_out = self.txt_mlp_out(txt_mlp_hidden).astype(encoder_hidden_states.dtype)
        encoder_hidden_states = encoder_hidden_states + txt_gate2[:, None, :] * txt_mlp_out

        return encoder_hidden_states, hidden_states


class QwenTransformerBlockApplier:
    @staticmethod
    def apply_from_handler(module: QwenTransformerBlockMLX, block_weights: dict) -> None:
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

    @staticmethod
    def verify_weights(module: QwenTransformerBlockMLX, block_weights: dict, print_first_values: bool = False) -> dict:
        import numpy as _np

        results = {
            "matched": [],
            "mismatched_shape": [],
            "missing_in_weights": [],
            "unused_weight_keys": [],
            "consumed_keys": [],
        }

        def fmt(arr: mx.array) -> str:
            return f"shape={tuple(arr.shape)} dtype={arr.dtype}"

        def record(path: str, mod_arr: mx.array, w_arr: mx.array):
            results["consumed_keys"].append(path)
            if tuple(mod_arr.shape) == tuple(w_arr.shape):
                msg = f"OK  {path}: {fmt(w_arr)}"
                if print_first_values:
                    try:
                        mv = _np.array(mod_arr).reshape(-1)[:3]
                        wv = _np.array(w_arr).reshape(-1)[:3]
                        msg += (
                            f" mod[:3]={_np.array2string(mv, precision=6)}, w[:3]={_np.array2string(wv, precision=6)}"
                        )
                    except Exception:
                        pass
                results["matched"].append(msg)
            else:
                results["mismatched_shape"].append(
                    f"SHAPE MISMATCH {path}: module {fmt(mod_arr)} vs weight {fmt(w_arr)}"
                )

        # mod linears
        if "img_mod" in block_weights and "1" in block_weights["img_mod"]:
            lin = block_weights["img_mod"]["1"]
            if "weight" in lin:
                record("img_mod.1.weight", module.img_mod_linear.weight, lin["weight"])
            else:
                results["missing_in_weights"].append("img_mod.1.weight")
            if "bias" in lin:
                record("img_mod.1.bias", module.img_mod_linear.bias, lin["bias"])
            else:
                results["missing_in_weights"].append("img_mod.1.bias")
        else:
            results["missing_in_weights"].extend(["img_mod.1.weight", "img_mod.1.bias"])

        if "txt_mod" in block_weights and "1" in block_weights["txt_mod"]:
            lin = block_weights["txt_mod"]["1"]
            if "weight" in lin:
                record("txt_mod.1.weight", module.txt_mod_linear.weight, lin["weight"])
            else:
                results["missing_in_weights"].append("txt_mod.1.weight")
            if "bias" in lin:
                record("txt_mod.1.bias", module.txt_mod_linear.bias, lin["bias"])
            else:
                results["missing_in_weights"].append("txt_mod.1.bias")
        else:
            results["missing_in_weights"].extend(["txt_mod.1.weight", "txt_mod.1.bias"])

        # attention
        attn = block_weights.get("attn", {})

        def chk_lin(prefix: str, target: nn.Linear, src: dict):
            if "weight" in src:
                record(f"{prefix}.weight", target.weight, src["weight"])
            else:
                results["missing_in_weights"].append(f"{prefix}.weight")
            if "bias" in src:
                record(f"{prefix}.bias", target.bias, src["bias"])
            else:
                results["missing_in_weights"].append(f"{prefix}.bias")

        if "to_q" in attn:
            chk_lin("attn.to_q", module.to_q, attn["to_q"])
        else:
            results["missing_in_weights"].extend(["attn.to_q.weight", "attn.to_q.bias"])
        if "to_k" in attn:
            chk_lin("attn.to_k", module.to_k, attn["to_k"])
        else:
            results["missing_in_weights"].extend(["attn.to_k.weight", "attn.to_k.bias"])
        if "to_v" in attn:
            chk_lin("attn.to_v", module.to_v, attn["to_v"])
        else:
            results["missing_in_weights"].extend(["attn.to_v.weight", "attn.to_v.bias"])
        if "add_q_proj" in attn:
            chk_lin("attn.add_q_proj", module.add_q_proj, attn["add_q_proj"])
        else:
            results["missing_in_weights"].extend(["attn.add_q_proj.weight", "attn.add_q_proj.bias"])
        if "add_k_proj" in attn:
            chk_lin("attn.add_k_proj", module.add_k_proj, attn["add_k_proj"])
        else:
            results["missing_in_weights"].extend(["attn.add_k_proj.weight", "attn.add_k_proj.bias"])
        if "add_v_proj" in attn:
            chk_lin("attn.add_v_proj", module.add_v_proj, attn["add_v_proj"])
        else:
            results["missing_in_weights"].extend(["attn.add_v_proj.weight", "attn.add_v_proj.bias"])

        if "norm_q" in attn and "weight" in attn["norm_q"]:
            record("attn.norm_q.weight", module.norm_q.weight, attn["norm_q"]["weight"])
        else:
            results["missing_in_weights"].append("attn.norm_q.weight")
        if "norm_k" in attn and "weight" in attn["norm_k"]:
            record("attn.norm_k.weight", module.norm_k.weight, attn["norm_k"]["weight"])
        else:
            results["missing_in_weights"].append("attn.norm_k.weight")
        if "norm_added_q" in attn and "weight" in attn["norm_added_q"]:
            record("attn.norm_added_q.weight", module.norm_added_q.weight, attn["norm_added_q"]["weight"])
        else:
            results["missing_in_weights"].append("attn.norm_added_q.weight")
        if "norm_added_k" in attn and "weight" in attn["norm_added_k"]:
            record("attn.norm_added_k.weight", module.norm_added_k.weight, attn["norm_added_k"]["weight"])
        else:
            results["missing_in_weights"].append("attn.norm_added_k.weight")

        if "to_out" in attn and len(attn["to_out"]) > 0:
            to_out0 = attn["to_out"][0]
            chk_lin("attn.to_out[0]", module.attn_to_out[0], to_out0)
        else:
            results["missing_in_weights"].extend(["attn.to_out[0].weight", "attn.to_out[0].bias"])
        if "to_add_out" in attn:
            chk_lin("attn.to_add_out", module.to_add_out, attn["to_add_out"])
        else:
            results["missing_in_weights"].extend(["attn.to_add_out.weight", "attn.to_add_out.bias"])

        # mlps
        img_mlp = block_weights.get("img_mlp", {}).get("net", {})
        if "0" in img_mlp and "proj" in img_mlp["0"]:
            chk_lin("img_mlp.net.0.proj", module.img_mlp_in, img_mlp["0"]["proj"])
        else:
            results["missing_in_weights"].extend(["img_mlp.net.0.proj.weight", "img_mlp.net.0.proj.bias"])
        if "2" in img_mlp:
            chk_lin("img_mlp.net.2", module.img_mlp_out, img_mlp["2"])
        else:
            results["missing_in_weights"].extend(["img_mlp.net.2.weight", "img_mlp.net.2.bias"])

        txt_mlp = block_weights.get("txt_mlp", {}).get("net", {})
        if "0" in txt_mlp and "proj" in txt_mlp["0"]:
            chk_lin("txt_mlp.net.0.proj", module.txt_mlp_in, txt_mlp["0"]["proj"])
        else:
            results["missing_in_weights"].extend(["txt_mlp.net.0.proj.weight", "txt_mlp.net.0.proj.bias"])
        if "2" in txt_mlp:
            chk_lin("txt_mlp.net.2", module.txt_mlp_out, txt_mlp["2"])
        else:
            results["missing_in_weights"].extend(["txt_mlp.net.2.weight", "txt_mlp.net.2.bias"])

        return results
