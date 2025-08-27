from __future__ import annotations

from dataclasses import dataclass
import math

import mlx.core as mx
import numpy as np
from mlx import nn

from mflux.models.flux.model.flux_transformer.ada_layer_norm_continuous import AdaLayerNormContinuous
from mflux.models.qwen.model.qwen_transformer.qwen_rope import QwenEmbedRopeMLX
from mflux.models.qwen.model.qwen_transformer.qwen_transformer_block import QwenTransformerBlockMLX


@dataclass
class QwenTransformerDims:
    in_channels: int = 64
    inner_dim: int = 3072
    joint_attention_dim: int = 3584
    timestep_proj_dim: int = 256


class QwenTimesteps(nn.Module):
    """Sinusoidal time projection matching Diffusers Timesteps(256, flip, scale=1000)."""

    def __init__(self, proj_dim: int = 256, scale: float = 1000.0):
        super().__init__()
        self.proj_dim = proj_dim
        self.scale = scale

    def __call__(self, timesteps: mx.array) -> mx.array:
        # Create positional embeddings of size proj_dim using sine/cosine
        half_dim = self.proj_dim // 2
        max_period = 10000.0
        exponent = -mx.log(mx.array(max_period)) * mx.arange(0, half_dim, dtype=mx.float32)
        exponent = exponent / (half_dim - 0.0)
        freqs = mx.exp(exponent)  # [half_dim]
        emb = timesteps.astype(mx.float32)[:, None] * freqs[None, :]
        emb = self.scale * emb
        emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
        # flip_sin_to_cos=True -> [cos, sin]
        emb = mx.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)
        return emb


class QwenTimestepEmbedding(nn.Module):
    """Two-layer MLP with SiLU to map 256 -> inner_dim -> inner_dim."""

    def __init__(self, proj_dim: int, inner_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(proj_dim, inner_dim)
        self.linear_2 = nn.Linear(inner_dim, inner_dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.silu(self.linear_1(x))
        x = self.linear_2(x)
        return x


class QwenTimeTextEmbedMLX(nn.Module):
    """Combines Timesteps and TimestepEmbedding like Diffusers QwenTimestepProjEmbeddings."""

    def __init__(self, dims: QwenTransformerDims):
        super().__init__()
        self.time_proj = QwenTimesteps(proj_dim=dims.timestep_proj_dim)
        self.timestep_embedder = QwenTimestepEmbedding(proj_dim=dims.timestep_proj_dim, inner_dim=dims.inner_dim)

    def __call__(self, timestep: mx.array, hidden_states: mx.array) -> mx.array:
        time_proj = self.time_proj(timestep)
        time_emb = self.timestep_embedder(time_proj.astype(hidden_states.dtype))
        return time_emb


class QwenTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        patch_size: int = 2,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.attention_head_dim = attention_head_dim
        self.num_attention_heads = num_attention_heads
        self.joint_attention_dim = joint_attention_dim
        self.patch_size = patch_size

        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim

        # Input projections
        self.img_in = nn.Linear(in_channels, inner_dim)
        self.txt_norm = nn.RMSNorm(joint_attention_dim, eps=1e-6)
        self.txt_in = nn.Linear(joint_attention_dim, inner_dim)

        # Time/Text embedding
        self.time_text_embed = QwenTimeTextEmbedMLX(QwenTransformerDims(in_channels, inner_dim, joint_attention_dim))

        # RoPE helper
        self.pos_embed = QwenEmbedRopeMLX(theta=10000, axes_dim=[16, 56, 56], scale_rope=True)

        # Blocks
        self.transformer_blocks: list[QwenTransformerBlockMLX] = [
            QwenTransformerBlockMLX(dim=inner_dim, num_heads=num_attention_heads, head_dim=attention_head_dim)
            for _ in range(num_layers)
        ]

        # Output head
        self.norm_out = AdaLayerNormContinuous(inner_dim, inner_dim)
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * out_channels)

    def __call__(
        self,
        t: int,
        config,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        encoder_hidden_states_mask: mx.array,
    ) -> mx.array:
        # Compute internal flux_transformer details (like Flux pattern)
        side = int(round(math.sqrt(hidden_states.shape[1])))
        img_shapes = [(1, side, side)]
        txt_seq_lens = [int(mx.sum(encoder_hidden_states_mask[i]).item()) for i in range(encoder_hidden_states_mask.shape[0])]

        # Resolve timestep from t and config (like Flux)
        timestep_value = config.get_qwen_timestep(t)
        batch = hidden_states.shape[0]
        timestep = mx.array(np.full((batch,), timestep_value, dtype=np.float32))

        hs = self.img_in(hidden_states)
        txt = self.txt_in(self.txt_norm(encoder_hidden_states))

        temb = self.time_text_embed(timestep, hs)

        # RoPE from computed shapes
        img_rot, txt_rot = self.pos_embed(video_fhw=img_shapes, txt_seq_lens=txt_seq_lens)

        # Blocks
        for idx, block in enumerate(self.transformer_blocks):
            # enable detailed debug for first, second and last blocks
            block.debug = idx in (0, 1, len(self.transformer_blocks) - 1)
            txt, hs = block(
                hidden_states=hs,
                encoder_hidden_states=txt,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=(img_rot, txt_rot),
            )

        # Output head (use only image stream)
        hs = self.norm_out(hs, temb)
        out = self.proj_out(hs)
        return out


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

    @staticmethod
    def verify_weights(module: QwenTransformer, weights: dict, print_first_values: bool = False) -> dict:
        """
        Verify that all handler weights map to named parameters in the MLX module.

        Returns a dict with lists: matched, mismatched_shape, missing_in_weights, unused_weight_keys.
        """
        import numpy as _np

        results = {
            "matched": [],
            "mismatched_shape": [],
            "missing_in_weights": [],
            "unused_weight_keys": [],
        }

        consumed = set()

        def fmt(arr: mx.array) -> str:
            return f"shape={tuple(arr.shape)} dtype={arr.dtype}"

        def record_match(path: str, mod_arr: mx.array, w_arr: mx.array):
            consumed.add(path)
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

        # Top-level
        if "img_in" in weights:
            wi = weights["img_in"]
            if "weight" in wi:
                record_match("img_in.weight", module.img_in.weight, wi["weight"])
            else:
                results["missing_in_weights"].append("img_in.weight")
            if "bias" in wi:
                record_match("img_in.bias", module.img_in.bias, wi["bias"])
            else:
                results["missing_in_weights"].append("img_in.bias")
        else:
            results["missing_in_weights"].extend(["img_in.weight", "img_in.bias"])

        if "txt_norm" in weights and "weight" in weights["txt_norm"]:
            record_match("txt_norm.weight", module.txt_norm.weight, weights["txt_norm"]["weight"])
        else:
            results["missing_in_weights"].append("txt_norm.weight")

        if "txt_in" in weights:
            wi = weights["txt_in"]
            if "weight" in wi:
                record_match("txt_in.weight", module.txt_in.weight, wi["weight"])
            else:
                results["missing_in_weights"].append("txt_in.weight")
            if "bias" in wi:
                record_match("txt_in.bias", module.txt_in.bias, wi["bias"])
            else:
                results["missing_in_weights"].append("txt_in.bias")
        else:
            results["missing_in_weights"].extend(["txt_in.weight", "txt_in.bias"])

        # Time/text embedder
        tte = weights.get("time_text_embed", {}).get("timestep_embedder", {})
        if tte:
            l1 = tte.get("linear_1", {})
            if "weight" in l1:
                record_match(
                    "time_text_embed.timestep_embedder.linear_1.weight",
                    module.time_text_embed.timestep_embedder.linear_1.weight,
                    l1["weight"],
                )
            else:
                results["missing_in_weights"].append("time_text_embed.timestep_embedder.linear_1.weight")
            if "bias" in l1:
                record_match(
                    "time_text_embed.timestep_embedder.linear_1.bias",
                    module.time_text_embed.timestep_embedder.linear_1.bias,
                    l1["bias"],
                )
            else:
                results["missing_in_weights"].append("time_text_embed.timestep_embedder.linear_1.bias")
            l2 = tte.get("linear_2", {})
            if "weight" in l2:
                record_match(
                    "time_text_embed.timestep_embedder.linear_2.weight",
                    module.time_text_embed.timestep_embedder.linear_2.weight,
                    l2["weight"],
                )
            else:
                results["missing_in_weights"].append("time_text_embed.timestep_embedder.linear_2.weight")
            if "bias" in l2:
                record_match(
                    "time_text_embed.timestep_embedder.linear_2.bias",
                    module.time_text_embed.timestep_embedder.linear_2.bias,
                    l2["bias"],
                )
            else:
                results["missing_in_weights"].append("time_text_embed.timestep_embedder.linear_2.bias")

        # Blocks
        blocks = weights.get("transformer_blocks", []) or []
        for i, bw in enumerate(blocks[: len(module.transformer_blocks)]):
            sub = QwenTransformerBlockApplier.verify_weights(module.transformer_blocks[i], bw, print_first_values)
            # Merge
            for k in results:
                if k in sub:
                    results[k].extend([f"[block.{i}] {msg}" for msg in sub[k]])
            # Track consumed keys from block verifier to figure out unused later
            for key in sub.get("consumed_keys", []):
                consumed.add(f"transformer_blocks[{i}].{key}")

        # Output head
        outw = weights.get("output", {})
        norm_out = outw.get("norm_out", {})
        if "linear.weight" in norm_out:
            record_match("output.norm_out.linear.weight", module.norm_out.linear.weight, norm_out["linear.weight"])
        else:
            results["missing_in_weights"].append("output.norm_out.linear.weight")
        if "linear.bias" in norm_out:
            record_match("output.norm_out.linear.bias", module.norm_out.linear.bias, norm_out["linear.bias"])
        else:
            results["missing_in_weights"].append("output.norm_out.linear.bias")
        if "proj_out" in outw:
            po = outw["proj_out"]
            if "weight" in po:
                record_match("output.proj_out.weight", module.proj_out.weight, po["weight"])
            else:
                results["missing_in_weights"].append("output.proj_out.weight")
            if "bias" in po:
                record_match("output.proj_out.bias", module.proj_out.bias, po["bias"])
            else:
                results["missing_in_weights"].append("output.proj_out.bias")

        # Compute unused keys by flattening handler dict
        def flatten(d, prefix=""):
            items = []
            if isinstance(d, dict):
                for k, v in d.items():
                    items.extend(flatten(v, f"{prefix}.{k}" if prefix else k))
            elif isinstance(d, list):
                for idx, v in enumerate(d):
                    items.extend(flatten(v, f"{prefix}[{idx}]"))
            else:
                items.append(prefix)
            return items

        all_leaf_keys = set(flatten(weights))
        # We matched at higher level paths like img_in.weight etc. Keep only leaf keys that correspond to tensors
        results["unused_weight_keys"] = sorted([k for k in all_leaf_keys if k not in consumed])

        # Pretty print summary
        print(
            f"Weight verification summary: matched={len(results['matched'])}, mismatched_shape={len(results['mismatched_shape'])}, missing_in_weights={len(results['missing_in_weights'])}, unused_weight_keys={len(results['unused_weight_keys'])}"
        )
        for msg in results["mismatched_shape"]:
            print(msg)
        if results["missing_in_weights"]:
            print("Missing weight entries:")
            for m in results["missing_in_weights"]:
                print(f"  - {m}")
        if results["unused_weight_keys"]:
            print("Unused handler keys (not applied):")
            for m in results["unused_weight_keys"]:
                print(f"  - {m}")

        return results
