## Qwen Image Port: Hard‑Won Truths and Techniques

This document consolidates the essential, reusable lessons from porting the Qwen Image VAE decoder to MLX. It omits block-by-block stories, numeric journeys, and transient scripts. Use this as the playbook for the transformer and text encoder ports.

### Core principles
- **Deterministic first**: fixed seeds, identical inputs, identical code paths
- **Find the first divergence**: layer-by-layer comparison before any refactors
- **Architecture over precision**: confirm structure and dataflow before blaming dtype
- **Ground truth = raw weights**: inspect the pretrained file’s keys; do not trust assumptions

### Reference-side workflow (PyTorch)
- Run the diffusers reference in a separate repo with fixed seed and fixed resolution
- **Modify production code, not test scripts**: Add prints/breakpoints directly in the real forward pass rather than creating separate debug scripts that may contain bugs
- Export named tensors for MLX ingestion; use clear, stable names (e.g., `vae/pre_decode_latents.pt`, `vae/up_block_1/res_0_out.pt`)
- Keep all run conditions repeatable; store the exact prompt/settings alongside tensors
- **Trace active code paths**: Add strategic prints to see which branches actually execute (dropout, conditional blocks, etc.)

### Minimal workflow (repeatable)
0) **Prepare reference tensors and run conditions**
- Generate deterministic inputs/outputs from the reference implementation and save them to disk
- Document the exact prompt/seed/size/steps used to create them

1) **Inspect weights (ground truth)**
- List relevant keys and shapes to learn true component presence and naming
- Derive expected shapes and which submodules actually exist

2) **Load and verify weights**
- Match MLX parameter paths exactly; MLX update is silent on missing paths
- Provide Python lists where modules expose lists; avoid dicts with numeric string keys
- Detect silent failures quickly: all‑zero biases, unreasonable mean/std, odd dtypes/shapes
- Re-load twice; values must be identical across runs (no randomness)
- Spot-check specific parameters against the reference (compare a few leading values for `conv_in`, early resblocks, `conv_out`)

3) **Component boundaries, then end‑to‑end**
- Hook key boundaries and compare with strict tolerances; only advance once clean
- Prefer direct hooks in production forward passes over separate test code to avoid drift
- **Chain debugging is critical**: In multi-step components (A→B→C), if step A is wrong, B and C will compound the error
- Trace chains A→B→C with named capture points so you can see exactly where agreement turns into divergence
- **Find the first divergence point**: Don't proceed until you've identified and fixed the earliest disagreement

### Practical tolerances
- **1e‑6**: numerical identity (unit tests)
- **1e‑5 to 1e‑4**: high precision, typical target
- **≈1e‑3**: acceptable if not on a critical accumulation path; may still produce visually identical images
- **>1e‑2**: treat as a bug unless justified by dtype differences
- **Context matters**: Reference 1.33 vs MLX 1.34 can be acceptable depending on the operation and position in the pipeline
- **Final arbiter**: Visual image comparison often reveals that seemingly "large" tensor differences produce nearly identical outputs

### Weight mapping essentials
- **Plot full hierarchy**: Print/inspect the complete weight dictionary structure from pretrained files; this must exactly match your MLX module hierarchy
- **Exact path matching**: A single level mismatch in the hierarchy will cause silent weight loading failures, leaving random weights
- **Lists vs dicts**: list modules need actual Python lists, not dicts of "0", "1", ...
- **No double nesting**: don't introduce containers (e.g., `conv3d`) twice in different layers
- **Verify weight assignment**: After loading, spot-check actual parameter values (first few numbers) against reference implementation to confirm weights were set correctly
- **Fast health checks**: shapes/dtypes; bias non‑zero; weight std within expected ranges; identical values across multiple runs (no randomness)

### Required layout conversions
- Conv2D: PyTorch `(out, in, h, w)` → MLX `(out, h, w, in)` via transpose `(0, 2, 3, 1)`
- Conv3D: PyTorch `(out, in, d, h, w)` → MLX `(out, d, h, w, in)` via transpose `(0, 2, 3, 4, 1)`

### MLX implementation patterns that prevent bugs
- **Channels‑last kernels**: wrap convs with transpose in/out; keep external tensors consistent
- **3D causal convs**: handle tuple params explicitly (padding, kernel_size, stride); don’t assume ints
- **Upsampling paths**: single‑frame image paths may bypass temporal convs; mirror reference behavior

### Determinism and seeds across frameworks
- **Critical principle**: Identical inputs matter infinitely more than identical seeds; RNGs differ across frameworks
- **When importing tensors**: If you import a tensor from diffusers into MLX, MLX-side seeds become completely irrelevant since randomness already occurred in the reference
- **Seed confusion trap**: Setting MLX seed to 42 because diffusers used 42 is meaningless—different RNGs will produce completely different results
- **Compare the same thing**: Ensure you're comparing truly identical operations, not similar-looking but different computations
- Only set seeds where randomness actually occurs (e.g., noise sampling in the reference run)

### Minimal tooling to keep
- Layer‑by‑layer comparison: `debug_layer_by_layer.py`
- End‑to‑end parity from saved latents: `test_end_to_end_image.py`

Examples:
```bash
MFLUX_SILENCE_TRACE=1 uv run python -u debug_layer_by_layer.py
uv run python -u test_end_to_end_image.py
```

### Final acceptance: image-level parity
- Tensor diffs of ~1e-3 can be acceptable if the final image is visually identical
- Use image comparison as the final arbiter once intermediate checks look reasonable

### What to omit going forward (on purpose)
- Block‑specific anecdotes and long metric journeys
- Per‑script narratives and environment logs
- Techniques that solved one‑offs we will not reuse

### Compact checklist (reuse every time)
- Inspect raw weight keys; note shapes and true hierarchy
- Define exact MLX parameter paths; use lists where needed
- Implement explicit transpositions for all conv weights (2D/3D)
- After loading, spot‑check: shapes, dtypes, means/stds, non‑zero biases
- Hook critical boundaries; compare tensors with strict tolerances
- Fix the first divergence; re‑run; repeat
- Keep inputs and random states identical across frameworks

### Success criteria (per stage)
- Early layers (post‑quant, conv_in): ≤1e‑2 max diff, stable ranges
- Complex blocks: ≤1e‑3 to 1e‑4 where practical; stable downstream behavior
- End‑to‑end: visually equivalent images from identical latents/settings

### Notes for upcoming phases
- Apply this exact playbook to transformer and text encoder
- Keep this doc small; add only patterns that will matter again

### Trace real code paths in the reference
- **Critical for complex implementations**: Diffusers code has many conditional branches that may not be active
- Add strategic prints in the reference forward pass to see which branches actually execute (e.g., dropout checks, temporal convolutions)
- **Inspect what's actually called**: Use debug mode with breakpoints or print statements to trace execution flow
- **Port only active paths**: Don't assume all code in the reference needs porting; focus on paths that actually run for your test case
- Skip unused branches until needed; this avoids porting dead code that could introduce bugs

### Working style that scales
- **Take extremely small steps**: Porting large chunks inevitably introduces multiple failure points that become very hard to debug
- **Single source of truth**: Modify the production forward pass with temporary prints/hooks rather than creating parallel test harnesses that may have their own bugs
- **Script reliability matters**: Debug scripts with bugs will lead you to wrong conclusions; prefer modifying known-good production code
- **Clear problem definition**: Always know exactly what is currently working, what is the next small problem to solve, and what can be safely skipped
- Remove temporary prints once validated to keep the codebase clean
- **Avoid confusion**: Don't keep multiple debug approaches active simultaneously; focus on one small, well-defined task

This single document supersedes prior separate READMEs and summaries. If a section becomes overly specific, prune it; if a new lesson recurs, add it briefly here.
