### Debugger & Tooling Improvement Ideas

This document captures rough edges and potential improvements for the `mflux_debugger` tools that became apparent while debugging the FIBO VAE decoder.

The goal: **make cross‑framework debugging smoother, faster, and less surprising for both humans and agents.**

---

### 1. CLI JSON / `continue` UX

**Observed issues**

- `mflux-debug-pytorch continue` sometimes reported:

  ```text
  ❌ Request failed: Expecting value: line 1 column 1 (char 0)
  ```

  even though the underlying program had progressed and tensors were being saved.
- When the program was “already running in background”, `continue` returned:

  ```text
  ⚠️ Execution already running in background
  ```

  without a clear way to learn whether it ultimately hit a breakpoint or just finished.

**Suggestions**

- Add a **non‑JSON / quiet mode** for `continue`, e.g.:

  ```bash
  mflux-debug-pytorch continue --no-json
  ```

  that:
  - Prints a concise one‑line status (`paused at X`, `finished`, `error Y`).
  - Avoids trying to parse partial JSON responses when the server logs extra text.
- Improve error reporting:
  - If JSON parsing fails, show:
    - “Server returned non‑JSON response – likely due to logging or crash.”
    - A hint to inspect `script_output.log` for details.
  - Don’t show a stack trace; use a clear, high‑level summary.
- When execution is already running:
  - Let `continue` **block until either completion or a new breakpoint**, instead of immediately returning a warning.
  - Or provide an explicit `status` poll at the same time (`continue --wait-status`).

This would make scripting around the CLI much more robust and reduce the confusing “Expecting value” noise.

---

### 2. Storage warnings & tensor directory management

**Observed issues**

- The tensor directory (`mflux_debugger/tensors/latest`) grew beyond 3 GB; repeated warnings like:

  ```text
  ⚠️  WARNING: Debug tensors directory is 3.67 GB (warning threshold: 1.0 GB)
  ```

  appeared many times in a single run.
- For automated agents this is mostly noise; the important bit is **“you’re about to hit disk issues”**, but the repeated messages clutter the output.

**Suggestions**

- Rate‑limit the warnings per run:
  - At most once per script invocation or once per N minutes.
  - Provide a single, clear recommendation to run `mflux-debug-clean` or `debug_clear()`.
- Provide a **configurable size limit** via env or CLI:
  - E.g. `MFLUX_DEBUG_MAX_TENSORS_GB=10`.
  - When exceeding the hard limit, fail fast with a clear error and instructions.
- Add a **simple CLI wrapper** for clearing only tensors:

  ```bash
  mflux-debug-clean --tensors-only
  ```

  so you don’t necessarily delete logs and coverage reports when you just need disk space.

---

### 3. Weight inspector quality‑of‑life improvements

The current CLI and API are already very useful, but a few additions would be especially helpful during porting:

**a. Mapping verification helpers**

- Provide a `--verify-mapping` mode that, given an HF key:

  ```bash
  mflux-debug-inspect-weights briaai/FIBO \
    --verify-mapping "decoder.up_blocks.0.resnets.0.conv1.weight"
  ```

  would show:
  - Expected MLX path from `WeightMapping`.
  - Actual MLX path (if present).
  - Transform applied (e.g. `transpose_conv3d_weight`).
  - Whether the raw vs mapped tensors match after transform (simple max‑abs diff).

This would eliminate a lot of manual reasoning when checking that mappings like `decoder.up_blocks.{block}.resnets.{res}.conv1.weight` → `decoder.up_blocks.{block}.resnets.{res}.conv1.conv3d.weight` are correct.

**b. Coverage analysis using real mappings**

- Implement the ideas in `WEIGHT_INSPECTOR_API.md`:
  - Pass a `mapping_class` (e.g. `FIBOWeightMapping`) into coverage analysis.
  - Report:
    - Which HF keys are expected to map but don’t.
    - Which MLX paths have no source in HF weights.
  - Optionally, flag suspicious patterns (e.g. many unmatched weights under a single component).

**c. Structure visualization**

- Add an easy way to print the **nested structure** of the mapped weights:

  ```bash
  mflux-debug-inspect-weights briaai/FIBO --component decoder --tree
  ```

  showing:
  - Lists vs dicts.
  - List lengths (`up_blocks[0..3]`, `resnets[0..2]`).
  - Tensor shapes under each path.

This would make it easier to confirm that the MLX module tree really mirrors the PyTorch one.

---

### 4. Debugger / script interaction notes

**Observed pattern**

- For heavy runs (like a full FIBO VAE decode), it was sometimes **simpler and more reliable** to:
  - Run the debug script directly via `uv run python debug_*.py`.
  - Let `debug_save` write all tensors.
  - Use `mflux-debug-mlx` or `uv run python - << 'EOF'` to inspect the saved tensors post‑hoc.

**Potential improvements**

- Expose a **“run once, then attach”** mode:
  - A CLI flag on the debug scripts (or a helper) that:
    - Runs the script to completion with `debug_save` active.
    - Stores a run ID.
    - Lets you later attach via `mflux-debug-mlx` / `mflux-debug-pytorch` using that run ID to inspect the tensors and checkpoints.
- Alternatively, add a simple “non‑interactive” flag to the debugger that:
  - Disables live JSON reporting.
  - Just runs the program while still honoring `debug_save` / `debug_checkpoint`.

This fits well with the workflow where you first collect all the tensors, then do a more exploratory inspection phase.

---

### 5. Minor ergonomics

- **TOML parsing warning with `uv`:**
  - `uv` complained about `[tool.uv.build-backend]` being unknown.
  - Not critical, but it’s noisy at the top of every `uv run` invocation.
  - Consider adjusting `pyproject.toml` or adding a small README note explaining the warning is safe to ignore.

- **Truncated terminal output in long runs:**
  - When very large amounts of output are produced, the terminal client truncated the middle.
  - Not a debugger bug per se, but a reminder:
    - Encourage users to run long debug sessions with `script_output.log` as the primary reference.
    - Possibly add a short section in the main README about “Where to look when the console truncates output”.

---

### 6. Summary

If you want to prioritize improvements that give the biggest payoff for debugging sessions like the FIBO VAE port, the top items are:

1. **Stabilize `continue` and status JSON handling**, or provide a `--no-json` / quiet mode.
2. **Tame storage warnings** (rate‑limit and add better cleanup options).
3. **Enhance the weight inspector** with:
   - Mapping verification.
   - Mapping‑aware coverage analysis.
   - Structure visualization.
4. **Support an explicit “run then inspect” workflow** where scripts run once and you inspect saved tensors after the fact.

These changes would make it significantly easier for both humans and future agents to debug cross‑framework ports with less friction and less guesswork.


