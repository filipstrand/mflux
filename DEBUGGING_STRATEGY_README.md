### FIBO VAE Decoder Debugging Strategy

This document captures the strategy that worked well to make the MLX FIBO VAE decoder numerically match the PyTorch `AutoencoderKLWan` decoder (as used in `BriaFiboPipeline`).

The goal: **treat PyTorch as ground truth, then systematically narrow down where MLX diverges, using the debugger instead of ad‑hoc scripts.**

---

### 1. Anchor early, then move the breakpoint forward

- Always start by finding checkpoints that are **guaranteed to be correct** and easy to compare:
  - Example pairs:
    - `*_vae_decode_input` (just before decoder)
    - `*_decoder_after_conv_in`
    - `*_decoder_after_mid_block`
- For each pair:
  - Compare shapes.
  - Compare a bunch of actual values (first 10 flattened, a few random slices), not just mean/std/min/max.
  - Only when these are **clearly matching** should you move your “before” checkpoint deeper into the model.

This prevents chasing issues that actually live earlier in the pipeline.

---

### 2. Reduce to the minimal set of checkpoints

Too many checkpoints make sessions noisy and slow. The pattern that worked:

- **Keep only:**
  - One **“before”** and one **“after”** at the level you’re investigating.
  - One **“before”** / **“after”** at any critical inner op (e.g. the resample Conv2d).
- **Remove or disable everything else** (or set `skip=True` so they only log to JSON without pausing).

Concretely, for the FIBO VAE decoder we used:

- Decoder level:
  - **Before:** `pytorch_decoder_after_mid_block` ↔ `mlx_decoder_after_mid_block`
  - **After:** `pytorch_decoder_after_up_block_0` ↔ `mlx_decoder_after_up_block_0`
- Resample Conv2d (inside `up_block_0`):
  - **Before:** `pytorch_resample_before_conv2d_compute` ↔ `mlx_resample_before_conv2d_compute`
  - **After:** `pytorch_resample_after_conv2d` ↔ `mlx_resample_after_conv2d`

If “before” matches but “after” differs, the bug is in that small slice of code.

---

### 3. Make PyTorch and MLX checkpoints symmetric

Whenever you add or keep a checkpoint:

- Ensure there is a **semantic twin** in the other framework:
  - Same position in the forward pass.
  - Same shape and layout (NHWC vs NCHW differences may require reshaping/transposing).
  - Ideally, a name that clearly pairs them (e.g. `pytorch_decoder_after_up_block_0` / `mlx_decoder_after_up_block_0`).
- Use `debug_save` on both sides so you can load the tensors with `debug_load` in a single environment (MLX REPL / CLI).

This symmetry is what allows fast comparisons without extra scripts.

---

### 4. Use the debugger as the comparison tool (avoid new scripts)

Prefer:

- `mflux-debug-mlx eval "..."` with `debug_load(...)` to:
  - Compute max absolute differences.
  - Inspect specific indices.
  - Print first 10–20 values to see qualitative alignment.
- Small `uv run python - << 'EOF'` snippets only when you need:
  - A more complex or multi-line comparison.
  - To avoid the debugger JSON wrapper issues.

Avoid:

- Standalone verification scripts that re‑implement parts of the pipeline.
- Ad‑hoc “one‑off” code that doesn’t use `debug_save`/`debug_load`.

The debugger already gives you the real tensors from the real pipelines; use it instead of reconstructing logic elsewhere.

---

### 5. Separate “inputs are the same” from “structure is the same”

When narrowing the bug:

1. **First confirm the inputs to the suspect region match**:
   - Example: `decoder_after_mid_block` was nearly identical in PyTorch and MLX.
2. **Then check any critical inner operation**:
   - Example: resample Conv2d output (`*_resample_after_conv2d`) was very close after fixing the Conv2d weight mapping.
3. **Only then blame the structure or higher‑level wiring**:
   - In our case, the big mismatch at `decoder_after_up_block_0` remained.
   - Inspecting class types and code revealed:
     - PyTorch uses `WanResidualUpBlock` with `DupUp3D` outer shortcut.
     - MLX only had a simple `WanUpBlock` with no outer shortcut.

This led directly to the right fix: implement `DupUp3D` + `WanResidualUpBlock` in MLX, not keep tweaking inner weights that were already correct in the mid block.

---

### 6. Use per‑residual checkpoints only when necessary

Per‑residual saves are expensive; only add them when the bug is clearly in a small stack:

- For `up_block_0` we added:
  - PyTorch: `pytorch_up_block_0_resnet_{k}_after_residual`
  - MLX: `mlx_up_block_0_resnet_{k}_after_residual`
- These are useful to answer:
  - “Which residual (0, 1, 2, …) is the first to diverge?”
  - If resnet 0 and 1 match but 2 doesn’t, focus debugging on resnet 2’s weights and wiring.

But don’t start with per‑residual checkpoints. Start with coarse “before/after” and drill down only when necessary.

---

### 7. Use weight inspector for name‑level and shape sanity, not as the primary weapon

The weight inspector is ideal for:

- Confirming that the HF side has the weights you expect:
  - Example: `decoder.up_blocks.0.resnets.0.conv1.weight` exists.
- Checking shapes and knowing which HF keys feed which MLX paths.

It is **not** the main bug‑finding tool in this workflow:

- Once earlier layers match numerically, global weight‑mapping mistakes become less likely.
- At that point, structural mismatches (missing blocks, missing shortcuts, wrong loops) are more probable.

Use the inspector to support conclusions, not as the first step.

---

### 8. Summary checklist for future debugging sessions

When porting or debugging another model:

1. **Pick a clear ground truth** (PyTorch or MLX) and treat it as canonical.
2. **Add symmetric checkpoints** in both implementations:
   - Global: entry / mid / exit of major blocks.
   - Local: critical ops (e.g. special resamples, attention blocks).
3. **Prune checkpoints** aggressively:
   - Keep one “before” + one “after” per region.
   - Disable or remove everything else.
4. **Compare tensors via debugger tools**, focusing on:
   - Shapes, a handful of indices, and max‑abs diff.
5. **Narrow the bug**:
   - If “before” matches and “after” doesn’t, bug is in between.
   - Drill down (e.g. per‑residual) only when needed.
6. **Check structure next**, then weights:
   - Are the same blocks present?
   - Are shortcuts and up/downsample factors implemented?
7. **Use weight inspector** to verify names/shapes and mapping coverage once you suspect mapping issues.

Following this pattern is what allowed us to turn a huge misalignment at `decoder_after_up_block_0` into a small (~1e‑1) numerical difference by identifying and porting the missing residual up‑block + shortcut structure.


