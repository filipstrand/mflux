from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from mflux.models.common.lora.layer.lokr_linear_layer import (
    LoKrLinear,
    reconstruct_lokr_delta,
)

# PEFT LoKr per-module parameter suffixes (Linear targets). Each factor is either
# full (``lokr_w1`` / ``lokr_w2``) or low-rank (``lokr_w*_a`` @ ``lokr_w*_b``). Order
# matters: match the longer ``_a`` / ``_b`` suffixes before the bare factor.
_LOKR_SUFFIXES = (
    ".lokr_w1_a",
    ".lokr_w1_b",
    ".lokr_w1",
    ".lokr_w2_a",
    ".lokr_w2_b",
    ".lokr_w2",
)


class LoKrLoader:
    """Apply a LoKr (LyCORIS Kronecker) adapter to an mflux transformer.

    LoKr cannot load through :class:`LoRALoader` — its keys are ``‹module›.lokr_w1`` /
    ``lokr_w2`` (full or low-rank ``_a``/``_b``) rather than ``lora_A``/``lora_B``, and
    the delta is a Kronecker product, not a low-rank product. The trained file (written
    by SceneWorks' ``write_lokr_adapter``) carries ``networkType=lokr`` plus ``alpha`` /
    ``rank`` in its safetensors metadata; the module path in each key is the bare
    diffusers/mflux module path (no ``transformer.``/``base_model.model.`` prefix), so it
    navigates the mflux module tree directly. Each matched Linear is replaced with a
    :class:`LoKrLinear` carrying the reconstructed delta (epic 2193: sc-2216/sc-2314).
    """

    @staticmethod
    def is_lokr(metadata: dict | None) -> bool:
        return str((metadata or {}).get("networkType", "")).strip().lower() == "lokr"

    @staticmethod
    def apply(
        transformer: nn.Module,
        weights: dict,
        metadata: dict,
        scale: float,
        *,
        role: str | None = None,
    ) -> tuple[int, set]:
        """Reconstruct + apply every LoKr-targeted module. Returns
        ``(applied_count, matched_keys)`` for the caller's match reporting."""
        rank = int(metadata.get("rank") or 1)
        # alpha defaults to rank (scaling 1.0) when absent, matching PEFT.
        alpha = float(metadata.get("alpha") or rank)

        # Group every lokr_* tensor by the module path that precedes the suffix.
        grouped: dict[str, dict[str, mx.array]] = {}
        matched_keys: set[str] = set()
        for key, value in weights.items():
            for suffix in _LOKR_SUFFIXES:
                if key.endswith(suffix):
                    module_path = key[: -len(suffix)]
                    factor_name = suffix[1:]  # e.g. "lokr_w1"
                    grouped.setdefault(module_path, {})[factor_name] = value
                    matched_keys.add(key)
                    break

        applied = 0
        for module_path, factors in grouped.items():
            if LoKrLoader._apply_to_module(transformer, module_path, factors, alpha, rank, scale, role):
                applied += 1
        return applied, matched_keys

    @staticmethod
    def _navigate(root: nn.Module, parts: list[str]):
        module = root
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            elif isinstance(module, dict) and part in module:
                module = module[part]
            else:
                module = getattr(module, part)
        return module

    @staticmethod
    def _apply_to_module(
        transformer: nn.Module,
        module_path: str,
        factors: dict[str, mx.array],
        alpha: float,
        rank: int,
        scale: float,
        role: str | None,
    ) -> bool:
        parts = module_path.split(".")
        try:
            module = LoKrLoader._navigate(transformer, parts)
        except (AttributeError, IndexError, KeyError, ValueError):
            print(f"   ⚠️  LoKr target not found in model: {module_path}")
            return False

        base = module.linear if isinstance(module, LoKrLinear) else module
        if not hasattr(base, "weight"):
            print(f"   ❌ LoKr target {module_path} is not a linear layer")
            return False

        delta = reconstruct_lokr_delta(
            alpha=alpha,
            rank=rank,
            base_shape=LoKrLinear.base_weight_shape(base),
            w1=factors.get("lokr_w1"),
            w1_a=factors.get("lokr_w1_a"),
            w1_b=factors.get("lokr_w1_b"),
            w2=factors.get("lokr_w2"),
            w2_a=factors.get("lokr_w2_a"),
            w2_b=factors.get("lokr_w2_b"),
        )
        lokr_layer = LoKrLinear.from_linear(base, delta=delta, scale=scale)
        lokr_layer._mflux_lora_role = role

        parent = LoKrLoader._navigate(transformer, parts[:-1])
        leaf = parts[-1]
        if leaf.isdigit():
            parent[int(leaf)] = lokr_layer
        elif isinstance(parent, dict) and leaf in parent:
            parent[leaf] = lokr_layer
        else:
            setattr(parent, leaf, lokr_layer)
        return True
