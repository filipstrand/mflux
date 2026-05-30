"""Real-weights validation of MLX-native LoKr on Z-Image-Turbo (sc-2216).

Loads the actual mflux Z-Image-Turbo transformer and applies a synthetic LoKr
through LoKrLoader, proving on real module paths/shapes that:
  (1) the loader navigates the real transformer tree (applied count > 0),
  (2) scale=0 is a bit-exact no-op on real (quantized) weights,
  (3) scale=1 changes the module output.

This is the production load path (LoKrLoader is what ZImage(lora_paths=...) calls
for a networkType=lokr file). A synthetic rank-1 (outer-product) factorization is
used so it works for any dims without a trained adapter; the "trained character
looks right" human-eval still needs a real LoKr training run.
"""

import mlx.core as mx

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.lora.layer.lokr_linear_layer import LoKrLinear, reconstruct_lokr_delta
from mflux.models.common.lora.mapping.lokr_loader import LoKrLoader
from mflux.models.z_image.variants.z_image import ZImage

TARGETS = [
    "layers.0.attention.to_q",
    "layers.0.attention.to_k",
    "layers.1.attention.to_v",
]


def _navigate(root, path):
    mod = root
    for part in path.split("."):
        mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
    return mod


def main():
    print("Loading Z-Image-Turbo (quantize=8) from cache ...")
    model = ZImage(quantize=8, model_config=ModelConfig.z_image_turbo())
    tr = model.transformer

    # Capture base modules + a reference output before mutation.
    bases = {p: _navigate(tr, p) for p in TARGETS}
    shapes = {p: LoKrLinear.base_weight_shape(m) for p, m in bases.items()}
    print("Real target shapes [out, in]:")
    for p, s in shapes.items():
        print(f"  {p}: {s}")

    xs = {p: mx.random.normal((2, s[1])) for p, s in shapes.items()}
    base_outs = {p: bases[p](xs[p]) for p in TARGETS}
    mx.eval(list(base_outs.values()))

    # Synthetic LoKr: rank-1 outer-product factors (w1 [out,1] ⊗ w2 [1,in] = [out,in]).
    weights = {}
    for p, (out_dim, in_dim) in shapes.items():
        weights[f"{p}.lokr_w1"] = mx.random.normal((out_dim, 1))
        weights[f"{p}.lokr_w2"] = mx.random.normal((1, in_dim))
    metadata = {"networkType": "lokr", "alpha": "8", "rank": "8"}

    # (1)+(2): apply at scale=0 → real-path navigation + bit-exact no-op.
    applied, matched = LoKrLoader.apply(tr, weights, metadata, scale=0.0)
    print(f"\nLoKrLoader.apply(scale=0): applied={applied}, matched={len(matched)}/{len(weights)} keys")
    assert applied == len(TARGETS), f"expected {len(TARGETS)} modules applied, got {applied}"

    for p in TARGETS:
        mod = _navigate(tr, p)
        assert isinstance(mod, LoKrLinear), f"{p} was not replaced with LoKrLinear"
        noop = mod(xs[p])
        mx.eval(noop)
        if not mx.array_equal(noop, base_outs[p]):
            raise SystemExit(f"❌ scale=0 NOT a no-op at {p}: max|Δ|={float(mx.max(mx.abs(noop - base_outs[p]))):.3e}")
    print("✅ (1) navigated real Z-Image module tree; (2) scale=0 is a bit-exact no-op on all targets")

    # (3): a scale=1 LoKrLinear on the original base must change the output.
    changed = 0
    for p in TARGETS:
        out_dim, in_dim = shapes[p]
        delta = reconstruct_lokr_delta(
            alpha=8.0, rank=8, base_shape=(out_dim, in_dim),
            w1=weights[f"{p}.lokr_w1"], w2=weights[f"{p}.lokr_w2"],
        )
        active = LoKrLinear.from_linear(bases[p], delta=delta, scale=1.0)
        out = active(xs[p])
        mx.eval(out)
        if not mx.allclose(out, base_outs[p]):
            changed += 1
    assert changed == len(TARGETS), f"scale=1 changed only {changed}/{len(TARGETS)}"
    print(f"✅ (3) scale=1 changes the output on all {changed} targets")
    print("\nALL CHECKS PASSED — MLX-native LoKr applies on real Z-Image-Turbo weights.")


if __name__ == "__main__":
    main()
