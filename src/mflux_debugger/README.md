# Debugger for PyTorch and MLX

Interactive debugging system optimized for ML workloads. Designed for AI agents to step through code, inspect tensors, and verify model implementations in real-time.

**Version: 0.5.0** - Now with **native CLI commands**, interactive tutorials, semantic breakpoints, and automatic checkpoint logging!

---

## 🎓 Start Here: Interactive Tutorial

**Learn the debugger by doing - run the tutorial before reading further:**

```bash
mflux-debug-mlx tutorial      # For MLX debugging
mflux-debug-pytorch tutorial  # For PyTorch debugging
```

The tutorial teaches:
- ✅ Starting/stopping debugging sessions
- ✅ Using semantic breakpoints (`debug_checkpoint()`)
- ✅ Inspecting variables with `eval`
- ✅ Using `debug_save()` and `debug_load()` for cross-framework comparison
- ✅ Automatic checkpoint logging
- ✅ Automatic polling (polls every 2 seconds until breakpoint/completion)

**After completing the tutorial**, refer to this README for:
- [Command Reference](#command-reference) - Quick lookup of all commands
- [Advanced Features](#advanced-features-beyond-the-tutorial) - Conditional breakpoints, multi-line eval
- [Model Porting Workflow](#model-porting-workflow) - Profile & Prune, working backwards strategy
- [Automatic Checkpoint Logging](#automatic-checkpoint-logging) - Offline analysis with `jq`
- [Live Debugging Lessons](#live-debugging-lessons-learned) - Real war stories from model porting
- [Troubleshooting](#troubleshooting) - Solutions to common issues

---

## 📖 Command Reference

**To see all available commands:**

```bash
mflux-debug-mlx --help      # Show all MLX debugging commands
mflux-debug-pytorch --help  # Show all PyTorch debugging commands
```

The tutorial teaches you the most important commands interactively. Use `--help` anytime for a quick reference.

---

## 🚀 Advanced Features Beyond the Tutorial

### 1. Debug Tensors - PyTorch ↔ MLX
Save and load tensors between PyTorch and MLX implementations (covered in tutorial, details below).

> **⚠️ IMPORTANT:** Always add the import at the **top of your file** with your other imports, not inline!

**In PyTorch code (save):**
```python
# At the top of your file, with other imports:
from mflux_debugger.tensor_debug import debug_save

# Then use it anywhere in your code:
class MyModel:
    def forward(self, x):
        # Simple case - save once
        latents = self.prepare_latents(...)
        debug_save(latents, "initial_latents", exit_after=True)
        
        # Loop case - save at each iteration
        for block in range(num_blocks):
            for timestep in range(num_timesteps):
                hidden_states = self.blocks[block](hidden_states, t=timestep)
                debug_save(hidden_states, f"hidden_states_{block}_{timestep}")
```

**In MLX code (load):**
```python
# At the top of your file, with other imports:
from mflux_debugger.tensor_debug import debug_load

# Then use it anywhere in your code:
class MyModel:
    def forward(self, x):
        # Simple case - load once
        latents = debug_load("initial_latents")  # Override computation
        
        # Loop case - load matching iterations
        for block in range(num_blocks):
            for timestep in range(num_timesteps):
                hidden_states = self.blocks[block](hidden_states, t=timestep)
                # Override with PyTorch value - minimal code change!
                # Note: debug_load() raises FileNotFoundError if tensor not found (fail-fast)
                hidden_states = debug_load(f"hidden_states_{block}_{timestep}")
```

**CLI commands:**
```bash
# List all debug tensors
mflux-debug-mlx tensors-list

# Get info about a specific tensor
mflux-debug-mlx tensors-info hidden_states_0_5

# Clear tensors
mflux-debug-mlx tensors-clear --name hidden_states_0_5  # Specific tensor
mflux-debug-mlx tensors-clear                            # All tensors (with confirmation)
```

**Storage limits and warnings:**
- **Warning threshold: 1 GB** - You'll see a warning when the entire `mflux_debugger/` directory exceeds 1 GB
- **Hard limit: 10 GB** - Saves are blocked and debugger won't start if `mflux_debugger/` exceeds 10 GB
- The system checks size automatically on every save and at debugger startup
- Use `mflux-debug-clean` to manage debug artifacts when needed

```python
# At the top of your file (or in a Python REPL/notebook):
from mflux_debugger.tensor_debug import debug_show_all, debug_clear

# Check current storage size
debug_show_all()  # Shows total size and lists all tensors

# Clean up when directory gets too large
debug_clear()  # Remove all tensors (with confirmation)
debug_clear(confirm=False)  # Skip confirmation
```

**Kill all processes (get a clean slate):**

Sometimes you start multiple debugging sessions, training runs, or inference scripts and just want to kill everything to start fresh:

```bash
# Preview what would be killed
mflux-debug-kill-all --dry-run

# Kill all MLX and PyTorch processes (with confirmation)
mflux-debug-kill-all

# Kill only MLX processes
mflux-debug-kill-all --mlx-only

# Kill only PyTorch processes
mflux-debug-kill-all --pytorch-only

# Force kill without confirmation
mflux-debug-kill-all --force
```

This command scans all running Python processes and kills those using MLX or PyTorch libraries. Useful when you've lost track of what's running!

### 2. Multi-Line Eval with Imports

```bash
# Simple (from tutorial)
mflux-debug-mlx eval "latents.shape"

# Advanced - multi-line with imports
mflux-debug-mlx eval "import torch
result = torch.randn(2, 3).sum()
result"
```

### 3. Conditional Breakpoints

```bash
mflux-debug-mlx break /path/file.py 123 --condition "timestep > 900"
```

---

## Table of Contents

1. **[🎓 Interactive Tutorial](#start-here-interactive-tutorial)** ⭐ **START HERE**
2. [📖 Command Reference](#command-reference) - Quick command lookup
3. [🚀 Advanced Features](#advanced-features-beyond-the-tutorial) - Conditional breakpoints, multi-line eval
4. [Installation](#installation) - Setup instructions
5. [Prerequisites: Editable Install Setup](#prerequisites-editable-install-setup)
6. [Model Porting Workflow](#model-porting-workflow) - Profile & Prune strategy
7. [Automatic Checkpoint Logging](#automatic-checkpoint-logging) - Offline analysis with `jq`
8. [Core Philosophy](#core-philosophy) - Interactive debugging over scripts
9. [MLX Lazy Evaluation](#mlx-lazy-evaluation-essential) - Understanding MLX behavior
10. [Model Porting Guide](#model-porting-guide) - Working backwards strategy
11. [Live Debugging Lessons Learned](#live-debugging-lessons-learned) - Real war stories
12. [Troubleshooting](#troubleshooting) - Common issues and solutions

---

## Installation

### Three-Part Installation

#### 1. Install MFLUX Repository (Development Mode)
```bash
cd ~/Desktop/mflux
make install
# This runs: uv pip install -e .
```

#### 2. Install Debugger CLI Tools (Global Commands)
```bash
cd ~/Desktop/mflux
uv tool install --editable .
```
This makes `mflux-debug-mlx`, `mflux-debug-pytorch`, `mflux-debug-clean`, and `mflux-debug-kill-all` available globally!

#### 3. Install Libraries in Editable Mode (Required for Breakpoints)
⚠️ **CRITICAL for PyTorch/Diffusers debugging:**

```bash
# Clone libraries
cd ~/Desktop
git clone https://github.com/huggingface/transformers.git
git clone https://github.com/huggingface/diffusers.git

# Install in editable mode
cd ~/Desktop/mflux
uv pip install -e ~/Desktop/transformers
uv pip install -e ~/Desktop/diffusers

# Verify (should show both as editable)
uv pip list --editable
```

**The debugger automatically checks this!** When you start `mflux-debug-pytorch`, it will verify editable installs and refuse to start if they're missing.

---

## Prerequisites: Editable Install Setup

### ⚠️ Why Editable Installs Are Required

**For model porting and debugging, you MUST have editable installs of `transformers` and `diffusers`.**

### Why This is Critical:

1. **Breakpoints won't work** - Python loads from `.venv/site-packages/` instead of your clones
2. **Profiler reports show wrong paths** - points to `.venv` files you can't edit
3. **Pruning is impossible** - you need direct access to the source files
4. **Can't iterate quickly** - changes require reinstalling packages

### Validation

The debugger **automatically validates** editable installs when you run `mflux-debug-pytorch start`. 

You can also manually check:
```bash
# Should show both transformers and diffusers
uv pip list --editable
```

**Validation Checklist:**
- ✅ `transformers` appears in `uv pip list --editable`
- ✅ `diffusers` appears in `uv pip list --editable`  
- ✅ Libraries are cloned locally (~/Desktop or elsewhere)
- ✅ `.git/` directories exist in both clones

**If validation fails,** the debugger will show clear instructions on how to fix it.

---

## Model Porting Workflow

**Now that you have editable installs, here's the model porting workflow:**

### Step 1: Profile & Prune the Reference Codebase

Use the **Profile & Prune** tool to discover and keep only the files actually executed:

```bash
# First-time setup: Create main-pruned branches
mflux-debug-prune setup

# Profile your reference script and prune unused files
mflux-debug-prune prune src/mflux_debugger/_scripts/debug_diffusers_txt2img.py

# Review the generated markdown report (mflux_debugger/prune/PROFILE_REPORT_*.md)
# Shows all files ranked by usage with categories

# If script breaks, restore files iteratively:
cd ~/Desktop/transformers
git checkout main -- src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py
git commit -m "Restore: models/qwen2_5_vl/modeling_qwen2_5_vl.py"

# Re-test and repeat until script works
# Then run prune again - it will keep the restored files
mflux-debug-prune prune src/mflux_debugger/_scripts/debug_diffusers_txt2img.py
```

**Learn the workflow:** Run `mflux-debug-prune tutorial` for an interactive guide!

**Key insight:** The tool profiles **actual execution** (not imports), so you only keep files that are genuinely used in the computation.

### 🎯 Use the Profile Report to Guide Breakpoint Placement

**The generated profile report is your roadmap for effective debugging!**

The report shows you:
- ✅ **Which files are actually executed** (not just imported)
- ✅ **Which implementation is used** when multiple exist (e.g., Qwen2_5_VL vs Qwen2VL)
- ✅ **Call frequency** to avoid breakpoints in hot loops
- ✅ **Full absolute paths** so you know exactly where files are located

**Example report excerpt:**
```
### 🧮 TEXT ENCODER
1. 🔥 `/Users/filip/Desktop/transformers/.../modeling_qwen2_5_vl.py` — 2,394 calls
2.    `/Users/filip/Desktop/transformers/.../modeling_qwen2.py` — 756 calls

### 🔗 PIPELINE  
1. 🔥 `/Users/filip/Desktop/diffusers/.../transformer_qwenimage.py` — 1,856 calls
```

**Why this matters:**
- ❌ **Without profiling:** You might set breakpoints in `modeling_qwen2_vl.py` when the code actually uses `modeling_qwen2_5_vl.py`
- ❌ **Without profiling:** You might breakpoint in a file with 10,000+ calls, generating massive trace files
- ✅ **With profiling:** You know exactly which files run, how often they're called, and their absolute paths

**Rule of thumb:**
- 🔥 **Files with >100 calls** - Set breakpoints at entry/exit points only (not in loops)
- **Files with <100 calls** - Safe to set breakpoints anywhere
- **Files not in report** - Won't execute, don't waste time setting breakpoints there!

### Step 2: Use the Debugger for Implementation Comparison

**Then come back here** to use the interactive debugger for comparing your MLX implementation against the PyTorch reference.



## Automatic Checkpoint Logging

**Tutorial Covered:** Basic `debug_checkpoint()` usage, `skip=True`, automatic JSON logging, and checkpoint inspection.

**This Section:** Advanced offline analysis patterns and strategic checkpoint placement.

### Analyzing Checkpoint Logs with jq

**✅ Best Practice: Use `jq` to analyze captured checkpoint JSON files instead of writing comparison scripts!**

After running your scripts with `debug_checkpoint()` calls, use `jq` to directly compare the JSON files:

#### Why This Approach is Better

**Advantages:**
- ✅ **Trustworthy** - Inspects actual captured runtime state, not reconstructed values
- ✅ **No missed differences** - Can't compare wrong variables or wrong execution points
- ✅ **Fast iteration** - Instant queries, no need to rerun scripts
- ✅ **Flexible** - Easy to drill into nested structures or compare specific fields
- ✅ **Version controlled** - Checkpoints are saved, can compare across commits

**Problems with comparison scripts:**
- ❌ Can check different execution points unintentionally
- ❌ May use different inputs or configurations
- ❌ Hard to maintain, accumulate as technical debt
- ❌ Can have bugs that hide real differences

#### Example: Comparing Variables Between PyTorch and MLX

```bash
# Compare latents shape and values from PyTorch and MLX runs
echo "=== PyTorch ===" && \
cat mflux_debugger/logs/runs/latest/my_pytorch_script/checkpoints/checkpoint_after_latents_*.json | \
jq -s 'first | .variables.latents | {shape, dtype, preview, stats}' && \
echo "" && \
echo "=== MLX ===" && \
cat mflux_debugger/logs/runs/latest/my_mlx_script/checkpoints/checkpoint_after_latents_*.json | \
jq -s 'first | .variables.latents | {shape, dtype, preview, stats}'
```

#### Example: Comparing Tensor Shapes Across Multiple Checkpoints

```bash
# Extract shapes from all checkpoints in a session
cat mflux_debugger/logs/runs/latest/my_script/checkpoints/checkpoint_*.json | \
jq '.variables | with_entries(select(.value.shape != null) | {key, value: .value.shape})'
```

#### Example: Finding Divergences

```bash
# Compare specific variable statistics across implementations
PT_MEAN=$(cat mflux_debugger/logs/runs/latest/pytorch_script/checkpoints/checkpoint_*.json | jq -r '.variables.hidden_states.stats.mean' | head -1)
MLX_MEAN=$(cat mflux_debugger/logs/runs/latest/mlx_script/checkpoints/checkpoint_*.json | jq -r '.variables.hidden_states.stats.mean' | head -1)

echo "PyTorch mean: $PT_MEAN"
echo "MLX mean: $MLX_MEAN"
```

#### Common jq Patterns for Checkpoint Analysis

```bash
# List all captured variables
jq '.variables | keys' checkpoint.json

# Get shapes of all tensor variables
jq '.variables | with_entries(select(.value.shape != null) | {key, value: .value.shape})' checkpoint.json

# Compare statistics for a specific variable
jq '.variables.hidden_states.stats' checkpoint.json

# Extract preview values (first 10 elements)
jq '.variables.latents.preview' checkpoint.json

# Find checkpoints where a condition is true
jq 'select(.variables.layer_num.value == 15)' checkpoint_*.json

# Compare across multiple files
jq -s 'map(.variables.cu_seqlens.shape) | unique' checkpoint_*.json
```

### Strategic Checkpoint Placement

**Key Rule:** Checkpoints log every execution. Poor placement → thousands of files. Use `skip=True` in loops.

**✅ GOOD Locations:**
- Pipeline entry/exit points (once per inference)
- Component boundaries (text encoder → transformer → VAE)
- Key state transitions (after latent creation, after scheduler step)

**❌ AVOID (or use skip=True):**
- Inner loops (28 transformer blocks = 2,890 files!)
- Frequently-called operations (use `skip=True`)

**Example:**
```python
# ✅ High-level - 6 checkpoint files
def generate(self, prompt, image):
    latents = self.prepare_latents()
    debug_checkpoint("after_latents", latents=latents)
    output = self.transformer(latents)
    debug_checkpoint("after_transformer", output=output)

# ✅ Loop with skip=True - logs but doesn't pause
for block in self.blocks:  # 28 iterations
    hidden_states = block(hidden_states)
    debug_checkpoint("block_out", hidden_states=hidden_states, skip=True)

# ✅ Conditional skip - pause only on specific iterations
for i, block in enumerate(self.blocks):
    hidden_states = block(hidden_states)
    # Only pause on first and last blocks, skip the rest
    debug_checkpoint(
        "block_out", 
        skip=(i != 0 and i != len(self.blocks) - 1),  # Skip middle blocks
        metadata={"block": i},
        hidden_states=hidden_states
    )
```

### Output Structure

```
mflux_debugger/
  logs/
    runs/
      latest/                      # Current runs (one per script)
        my_script/
          script_output.log        # Script stdout/stderr with checkpoint references
          checkpoints/             # Checkpoint JSON files
            checkpoint_after_latents_hit001.json
            checkpoint_transformer_forward_hit001.json
            checkpoint_scheduler_step_hit001.json
      archive/                     # Archived runs (moved when new session starts)
        my_script_20251106_150000/
          script_output.log
          checkpoints/
            checkpoint_*.json
    debugger/                      # Server logs
      mlx_debugger.log
      pytorch_debugger.log
  tensors/                         # debug_save() / debug_load() tensors
    input_tensor.npy
    hidden_states_0.npy
    archive/                       # Archived tensors (moved when new session starts)
      my_script_20251106_150000/
        input_tensor.npy
        hidden_states_0.npy
  images/                          # Generated images (if any)
```

### Example Checkpoint File Content

```json
{
  "checkpoint_name": "transformer_forward",
  "hit_count": 1,
  "timestamp": "2025-11-06T16:15:23.456789",
  "location": {
    "file": "pipeline_qwenimage_edit.py",
    "line": 837,
    "function": "__call__"
  },
  "code_context": {
    "start_line": 832,
    "end_line": 842,
    "lines": {
      "835": "                hidden_states = mx.concatenate([latents, static_image_latents], axis=1)",
      "837": "                noise = self.transformer(...)",
      "839": "                )"
    }
  },
  "variables": {
    "latents": {
      "type": "torch.Tensor",
      "shape": [1, 1024, 64],
      "dtype": "torch.float32",
      "preview": [-0.698, 1.429, -1.310, ...],
      "stats": {"min": -3.2, "max": 3.1, "mean": 0.02, "std": 1.0}
    },
    "prompt_embeds": {
      "type": "torch.Tensor",
      "shape": [1, 1385, 3584],
      "dtype": "torch.float16",
      "preview": [3.884, -0.190, 4.953, ...],
      "stats": {...}
    }
  }
}
```

Each file is **self-contained** - everything an AI agent needs to understand that execution point!


### Common Debugging Setup:

```bash
# Install commonly debugged libraries in editable mode
uv pip install -e ~/Desktop/diffusers     # For PyTorch reference implementations
uv pip install -e ~/Desktop/transformers  # For text encoders
uv pip install -e ~/Desktop/mflux         # For MFLUX itself

# Verify all are editable
uv pip list | grep -E "(diffusers|transformers|mflux)"
```

### Troubleshooting Path Issues:

**If breakpoints still don't hit:**

1. **Check which version is loaded:**
```python
# In your debug session, evaluate:
import diffusers
print(diffusers.__file__)
# Should point to: /Users/you/Desktop/diffusers/...
# NOT: .venv/lib/python3.XX/site-packages/diffusers/...
```

2. **Use absolute paths for breakpoints:**
```bash
# ✅ GOOD - Absolute path
mflux-debug-pytorch break /Users/you/Desktop/diffusers/src/diffusers/pipelines/flux/pipeline_flux.py 856

# ❌ BAD - Relative path (won't work)
mflux-debug-pytorch break ../diffusers/src/diffusers/pipelines/flux/pipeline_flux.py 856
```

3. **Restart your debug server after installing in editable mode:**
```bash
# Terminate and restart
mflux-debug-mlx terminate
mflux-debug-mlx start <your_script.py>
```

---

## Core Philosophy

### ⚠️ Interactive Debugging Over Scripts

**CRITICAL:** When investigating models or comparing implementations, **USE THE DEBUGGER**, not temporary scripts!

**Why:**
- ✅ Inspects actual production code
- ✅ Can't check the wrong thing
- ✅ No maintenance burden
- ✅ Git handles versioning
- ✅ Trustworthy results

**Problems with scripts:**
- ❌ Can compare different computation points
- ❌ Hard to maintain, accumulate over time
- ❌ Can be written incorrectly
- ❌ Create dual implementations that drift

**Only write scripts if explicitly requested.**

### Example: Comparing Implementations (Good Approach)

❌ **DON'T:**
```python
# Comparison script - might compare wrong things
def compare_encoders():
    mlx_enc = load_mlx()
    torch_enc = load_torch()
    # Easy to load wrong weights or compare wrong points
```

✅ **DO:**
```
# Debug Session 1 (PyTorch reference)
Set breakpoint at pipeline_flux.py line 828
Inspect prompt_embeds.shape  # [1, 512, 4096]

# Debug Session 2 (MLX implementation)  
Set breakpoint at flux.py line 72
Inspect prompt_embeds.shape  # Verify matches reference
```

Traces are automatically saved for offline comparison!

---

## MLX Lazy Evaluation (Essential!)

### The Challenge

**MLX uses lazy evaluation** - computations are deferred until `mx.eval()`:

```python
# This doesn't compute anything!
latents = mx.random.normal(shape=[1, 256, 64])

# Computation happens here:
mx.eval(latents)  # NOW it executes
```

**Impact on debugging:**
- ❌ Breakpoint BEFORE `mx.eval()` → tensor has no values
- ❌ `.mean()`, `.shape` → also lazy
- ❌ Debugger shows nothing or hangs

### Solution: Debugger Auto-Handles This!

**Good news:** The debugger **automatically handles MLX lazy evaluation**!

- ✅ Selectively traces only user code
- ✅ Skips ML library internals
- ✅ Evaluates tensors automatically when inspecting
- ✅ No manual `mx.eval()` needed for debugging

### Best Breakpoint Locations for MLX

✅ **GOOD** - After natural evaluation points:
```python
for t in time_steps:
    latents = self.transformer(...)
    latents = self.scheduler.step(...)
    mx.eval(latents)  # Already here for progress!
    # ✅ Set breakpoint here - tensor is evaluated
```

❌ **BAD** - Before evaluation:
```python
latents = mx.random.normal(...)
# ❌ Breakpoint here - lazy, not computed yet
mx.eval(latents)
# ✅ Breakpoint here - now computed
```

---

## Model Porting Guide

### Working Backwards Strategy (Proven Successful)

**Problem:** Entire pipeline is hard to debug at once.

**Solution:** Implement and verify components **backwards**, from output to input.

```
Traditional (hard):
Text → Latent → Transformer → VAE → Image
├─────────── implement everything ───────────┤
                   then debug

Backwards (easy):
Text ← Latent ← Transformer ← VAE ← Image
                              ↑
                         Start here!
```

**Implementation order:**

1. **VAE Decoder** (outputs images - easy to verify visually)
2. **Transformer last block** (verify numerically)
3. **Work backwards through transformer blocks**
4. **Text encoder**
5. **Latent creation**
6. **End-to-end check**

**Why this works:**
- Isolates each component
- Visual verification early (VAE produces images)
- Removes non-determinism by using saved inputs
- Builds confidence incrementally

### Verification Framework

**Tolerance levels:**
- `error < 1e-5` → ✅ Clearly correct
- `1e-5 < error < 0.1` → ⚠️ Check visual output
- `error > 0.1` → ❌ Clearly broken

**Don't reject implementations in the gray area without visual inspection!**

### Small Verified Chunks

✅ **Verify each component thoroughly before moving on**  
✅ **Don't accumulate uncertainty**  
✅ **Use Git to revert if needed**  
❌ **Don't doubt previously verified components**

---

## Live Debugging Lessons Learned

### Session 1: Path Normalization Bug (FIXED!)

**Problem:** Breakpoints not hitting in Diffusers pipeline

**Root cause:**
- Breakpoints stored with absolute paths: `/Users/.../diffusers/file.py`
- Frame paths were relative: `../diffusers/file.py`
- Mismatch → breakpoint never matched!

**Fix:** Normalize all paths to absolute before comparison

```python
# In check_breakpoint():
filename = str(Path(frame.f_code.co_filename).resolve())
```

**Result:** ✅ Breakpoints hit reliably!

### Session 2: Crash Resilience (IMPLEMENTED!)

**Problem:** Server crashed during debugging → lost all trace data

**Root cause:** Traces only saved on explicit `terminate()` call

**Fix:** Incremental auto-save after every step

**Features added:**
- ✅ Auto-save after each breakpoint/evaluation
- ✅ Emergency save on unexpected exit (`atexit` handler)
- ✅ Timestamped trace files (never overwrite)
- ✅ Automatic session cleanup on restart

**Result:** Never lose debugging progress, even on crash!

### Session 3: Loop Breakpoints (NOW MANAGEABLE!)

**Problem:** Breakpoint in denoising loop hit 4x (once per step)

**Solution:** Added breakpoint management API

**New endpoints:**
- `POST /debug/breakpoint/remove` - Remove specific breakpoint
- `GET /debug/breakpoints` - List all breakpoints
- `POST /debug/breakpoints/clear` - Clear all at once

**Workflow:**
```bash
# Hit breakpoint once
mflux-debug-mlx continue

# Inspect
mflux-debug-mlx eval "i"

# Remove so we don't hit it again
mflux-debug-mlx remove /path/to/file.py 944

# Continue to next breakpoint (skips remaining iterations)
mflux-debug-mlx continue
```

### Session 4: Tensor Shape Verification

**Finding:** PyTorch and MLX implementations have **identical tensor shapes**!

| Stage | Tensor | Shape | Match |
|-------|--------|-------|-------|
| Text encoding (T5) | prompt_embeds | `[1, 512, 4096]` | ✅ |
| Text encoding (CLIP) | pooled_embeds | `[1, 768]` | ✅ |
| Initial latents | latents | `[1, 256, 64]` | ✅ |
| Unpacked latents | latents | `[1, 16, 32, 32]` | ✅ |

**This proves the architecture is correctly ported!**

### Session 5: Trace Recording

**Feature:** Automatic trace recording for offline analysis

**What's captured:**
- All breakpoints set
- All execution steps with location, code context, call stack
- All evaluations with results
- Timestamps for performance analysis

**Trace format:**
```json
{
  "session_id": "20251024_133020",
  "script_path": "/path/to/debug_mflux.py",
  "start_time": "2025-10-24T13:30:20...",
  "end_time": "2025-10-24T13:32:02...",
  "breakpoints": [
    {"file": "...", "line": 144, "condition": null}
  ],
  "steps": [
    {
      "step": 1,
      "step_type": "continue_async",
      "location": {"file": "...", "line": 144, "function": "generate_image"},
      "code_context": {"before": [...], "current": [...], "after": [...]},
      "call_stack": [...],
      "variables": {...}
    },
    {
      "step": 2,
      "step_type": "evaluation",
      "expression": "latents.shape",
      "result": "[1, 16, 32, 32]"
    }
  ],
  "total_steps": 6
}
```

**Location:** `mflux_debugger/traces/debug_NAME_TIMESTAMP.json`

### Session 6: Editable Installation Discovery (CRITICAL!)

**Problem:** Breakpoints not hitting in external library code (diffusers, transformers)

**Root cause:** Python loading package from `.venv/lib/python3.XX/site-packages/` instead of local clone

**Symptoms:**
- Breakpoint set at `/Users/you/Desktop/diffusers/src/diffusers/file.py:856`
- But Python loads from `/Users/you/mflux/.venv/lib/python3.11/site-packages/diffusers/...`
- Path mismatch → debugger never pauses!

**The Fix:** Install external libraries in editable mode:
```bash
uv pip install -e ~/Desktop/diffusers
uv pip install -e ~/Desktop/transformers
uv pip install -e ~/Desktop/mflux  # Don't forget mflux itself!
```

**How to verify:**
```python
# In debug session, evaluate:
import diffusers
print(diffusers.__file__)
# ✅ Should show: /Users/you/Desktop/diffusers/...
# ❌ BAD: .venv/lib/.../site-packages/diffusers/...
```

**Alternative check:**
```bash
uv pip list | grep diffusers
# ✅ Good: diffusers  0.30.0  /Users/you/Desktop/diffusers
# ❌ Bad: diffusers  0.30.0
```

**Result:** This single change makes debugging external libraries possible! Without editable mode, you can only debug your own code effectively.

**Important:** Always restart the debug server after switching to editable mode:
```bash
pkill -f "uvicorn mflux_debugger"
uvicorn mflux_debugger.fastapi_server:app --host 127.0.0.1 --port 8000
```

### Session 7: Virtual Environment Best Practices

**Lesson:** Keep debugging environments clean and consistent

**Best practices:**
1. **One virtual environment per project being debugged:**
   ```bash
   # mflux environment (for debugging mflux)
   cd ~/Desktop/mflux
   python3 -m venv .venv
   source .venv/bin/activate
   uv pip install -e .
   uv pip install -e ~/Desktop/diffusers  # Reference implementation
   ```

2. **Always verify which environment is active:**
   ```bash
   which python
   # Should show: /Users/you/Desktop/mflux/.venv/bin/python
   ```

3. **Document your setup in a script:**
   ```bash
   # ~/Desktop/mflux/setup_debug_env.sh
   #!/bin/bash
   python3 -m venv .venv
   source .venv/bin/activate
   uv pip install -e .
   uv pip install -e ~/Desktop/diffusers
   uv pip install -e ~/Desktop/transformers
   uv pip install fastapi uvicorn
   ```

4. **Reset environment if things get weird:**
   ```bash
   cd ~/Desktop/mflux
   rm -rf .venv
   ./setup_debug_env.sh
   ```

**Common mistakes:**
- ❌ Installing packages in global Python
- ❌ Forgetting to activate venv before debugging
- ❌ Installing regular (non-editable) packages after editable ones
- ❌ Having multiple conflicting versions in different locations

### Session 8: Real-World Model Port Verification (QWEN IMAGE)

**Task:** Verify PyTorch→MLX port by comparing tensor values at key stages

**The typical request:**
> "I would like you to step through the main chunks of the model (text encoder, initial latent, transformer, VAE) for both diffusers and MLX. Compare tensor values to ensure correct porting."

**Challenge:** Random seed produces different initial latents across frameworks

#### Solution: Minimal Code Intervention (No Helper Scripts!)

**Step 1: Save reference latents from diffusers**

```python
# In /path/to/diffusers/src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py
# After line: latents = self.prepare_latents(...)

# TEMP: Save latents for debugging
np.save("/Users/you/Desktop/initial_latents_diffusers.npy", latents.cpu().float().numpy())
import sys; sys.exit(0)  # Exit immediately after saving
```

Run once to save, then comment out the exit:
```python
# TEMP: Save latents for debugging
np.save("/Users/you/Desktop/initial_latents_diffusers.npy", latents.cpu().float().numpy())
# import sys; sys.exit(0)  # (commented out for full debugging)
```

**Step 2: Load saved latents in MLX**

```python
# In /path/to/mflux/src/mflux/latent_creator/latent_creator.py
import numpy as np
from pathlib import Path

@staticmethod
def create(seed: int, height: int, width: int) -> mx.array:
    # TEMP: Load latents from diffusers for comparison
    latents_path = Path("/Users/you/Desktop/initial_latents_diffusers.npy")
    if latents_path.exists():
        latents_np = np.load(latents_path)
        return mx.array(latents_np)
    
    # Original random generation
    return mx.random.normal(...)
```

**Step 3: Debug both implementations interactively**

```bash
# Debug Diffusers (PyTorch)
mflux-debug-pytorch start src/mflux_debugger/examples/debug_diffusers.py

# Set breakpoints at key locations
mflux-debug-pytorch break /path/to/pipeline.py 623  # After text encoding
mflux-debug-pytorch break /path/to/pipeline.py 639  # After latents
mflux-debug-pytorch break /path/to/pipeline.py 765  # Before VAE

# Run and wait for breakpoint (automatic polling)
mflux-debug-pytorch continue

# Inspect at breakpoint
mflux-debug-pytorch eval "prompt_embeds.shape"
mflux-debug-pytorch eval "(latents.mean().item(), latents.std().item())"

# Continue to next breakpoint
mflux-debug-pytorch continue
```

Repeat for MLX implementation.

#### Results from Real Session

| Component | Diffusers | MLX | Match? |
|-----------|-----------|-----|--------|
| **Text Encoder** | [1, 14, 3584]<br>mean=-0.104, std=5.09 | [1, 14, 3584]<br>mean=-0.101, std=5.06 | ✅ |
| **Initial Latents** | [1, 256, 64]<br>*(saved to disk)* | [1, 256, 64]<br>*(loaded from disk)* | ✅ |
| **Latents before VAE** | [1, 16, 1, 32, 32]<br>mean=-0.052, std=1.07 | [1, 16, 32, 32]<br>mean=-0.028, std=0.56 | ⚠️ |

**Finding:** Text encoder matches perfectly. Denoising loop has divergence (transformer/scheduler issue).

#### Key Principles Demonstrated

1. **✅ Minimal code changes:** 2 lines to save, 3 lines to load
2. **✅ No helper scripts:** All debugging via API calls in terminal
3. **✅ No workspace clutter:** Temporary changes in source files (easily reverted)
4. **✅ Interactive inspection:** Live evaluation at each breakpoint
5. **✅ Comment after use:** Once saved, comment out exit line for full debugging
6. **✅ Direct comparison:** Same input latents ensure fair comparison

**Why this approach is better than writing scripts:**
- ❌ Scripts need maintenance and accumulate over time
- ❌ Scripts can compare wrong things or be written incorrectly
- ✅ Interactive debugging inspects actual production code
- ✅ Git makes it easy to revert temporary changes
- ✅ Follows "Core Philosophy" of the debugger

**Lesson:** When asked to compare implementations, use **minimal inline changes + interactive debugging**, not helper scripts!

### Session 9: Vision Tower Debugging - Systematic Component Isolation (QWEN IMAGE EDIT)

**Task:** Debug MLX vision tower divergence from PyTorch reference implementation

**Context:** After fixing patch embedding, vision blocks were still diverging significantly:
- Block 1 input: ✅ Perfect match
- Block 1 output: ❌ 1.52x divergence (PT: 20.78 vs MLX: 13.69)

#### Critical Debugging Strategy #1: Work Backwards from Known-Good State

**The Principle:**
> "Start from an earlier point where things agree, then progress forward until you see divergence."

**Why This Works:**
- Downstream errors compound, making it hard to find root cause
- Working backwards isolates the **first** divergence point
- Provides clear "before/after" comparison

**Real Example:**

```python
# Step 1: Confirm patch_embed output matches (it did!)
# Block input: PT=14.66, MLX=14.62 ✅ Match

# Step 2: Check block output
# Block output: PT=20.78, MLX=13.69 ❌ Diverged!

# Step 3: Divergence is WITHIN the block - isolate components
# Add debug code to VisionBlock.__call__:
x_input = x
normed1 = self.norm1(x)
attn_out = self.attn(normed1, position_embeddings, cu_seqlens)
x = x + attn_out  # ← Breakpoint here

normed2 = self.norm2(x)
mlp_out = self.mlp(normed2)
x = x + mlp_out  # ← And here

# Step 4: Compare at each stage
# After attention: PT=13.69, MLX=14.62 → Attention is fine!
# After MLP: PT=20.78, MLX=13.69 → MLP is the problem!

# Step 5: Inspect MLP output directly
# PT mlp_out range: 14.58
# MLX mlp_out range: 0.51 ← 28.8x ERROR FOUND!
```

**Result:** Found root cause in 30 minutes instead of hours of guessing.

#### Critical Debugging Strategy #2: Handle Framework Convention Differences Locally

**The Principle:**
> "When MLX and PyTorch have different conventions (like NCDHW vs NDHWC), do a local transpose, compute, and transpose back. Keep the reference layout everywhere else."

**Why This Matters:**
- MLX uses channels-last (NDHWC) for Conv3d
- PyTorch uses channels-first (NCDHW) for Conv3d
- **Naive conversion leads to subtle shape bugs**

**Wrong Approach (What We Fixed):**
```python
# ❌ BAD: Receiving NDHWC directly, applying Conv3d
def __call__(self, pixel_values):  # [N, T, H, W, C]
    return self.proj(pixel_values)  # Wrong! Spatial dimensions mismatched
```

**Correct Approach (What We Implemented):**
```python
# ✅ GOOD: Match PyTorch's data flow, transpose only for Conv3d
def __call__(self, hidden_states):
    batch_size = hidden_states.shape[0]
    
    # Input: flattened [num_patches, C*T*H*W] (match PyTorch)
    # Reshape to PyTorch format (NCDHW)
    hidden_states = hidden_states.reshape(
        batch_size, self.in_channels, self.temporal_patch_size, 
        self.patch_size, self.patch_size
    )
    
    # LOCAL TRANSPOSE for Conv3d only
    hidden_states = hidden_states.transpose(0, 2, 3, 4, 1)  # NCDHW → NDHWC
    
    # Compute
    output = self.proj(hidden_states)
    
    # Flatten back to match PyTorch output format
    return output.reshape(batch_size, self.embed_dim)
```

**Key Insight:** Keep your code's data layout matching the reference (PyTorch) in 90% of the codebase. Only do local transposes where framework APIs require it.

**Result:** Patch embedding divergence fixed - from 2.5% exact matches to 25% (10x improvement)!

#### Additional Critical Lessons Learned

##### Lesson 3: Component Isolation When Blocks Diverge

**Strategy:** When a composite block (like VisionBlock = norm + attention + MLP + residuals) diverges, break it down:

1. **Expose intermediate values:**
   ```python
   # Instead of:
   x = x + self.mlp(self.norm2(x))
   
   # Do:
   normed2 = self.norm2(x)
   mlp_out = self.mlp(normed2)
   x = x + mlp_out  # ← Can set breakpoint and inspect mlp_out
   ```

2. **Trace each component separately:**
   - Input to component (normed2)
   - Output from component (mlp_out)  
   - After residual add (x)

3. **Compare statistics, not just values:**
   - **Range** reveals amplification/attenuation issues
   - **Mean** reveals bias problems
   - **Min/Max** shows extreme value handling

**Real Finding:** MLX MLP output range was 0.51 vs PyTorch's 14.58 → **28.8x error!**

##### Lesson 4: Don't Assume Architectures Match - Verify Against Source

**The Bug:** MLX implemented VisionMLP as simple 2-layer:
```python
# What we HAD (wrong):
fc1 → gelu → fc2
```

**The Reality:** PyTorch uses GLU-style 3-layer:
```python
# What we NEEDED (correct):
down_proj(silu(gate_proj(x)) * up_proj(x))
```

**How We Found It:**
1. Noticed 28.8x MLP output divergence
2. Checked PyTorch source: `transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`
3. Found `Qwen2_5_VLMLP` class with `gate_proj`, `up_proj`, `down_proj`
4. Realized: **Gated Linear Unit ≠ Simple MLP!**

**Key Operations:**
- `gate_proj(x)` → apply silu activation
- `up_proj(x)` → no activation
- Element-wise multiplication: `gate * up`
- `down_proj(result)`

**Lesson:** When debugging architecture ports, **always verify against source**, especially for:
- Activation functions (silu vs gelu vs relu)
- Layer structure (2-layer vs 3-layer)
- Special operations (gating, element-wise multiplication)
- Normalization types (RMSNorm vs LayerNorm)

**Result:** After fixing, MLP output: MLX=14.55 vs PT=14.58 ✅ **Perfect match!**

##### Lesson 5: Distinguish Weight Loading vs Implementation Bugs

**Finding:** Weight loading was already correct!
- SafeTensors had: `visual.blocks.0.mlp.gate_proj.weight`, `up_proj.weight`, `down_proj.weight`
- Weight handler was loading all three correctly

**But:** MLX code was only using 2 of them (fc1, fc2)!

**Lesson:** When debugging divergence, check **both**:
1. ✅ Are weights loaded? (Check weight handler)
2. ✅ Are weights used correctly? (Check model implementation)

**How to Verify:**
```python
# Check if weights exist
from safetensors import safe_open
with safe_open(model_file, framework='numpy') as f:
    mlp_keys = [k for k in f.keys() if 'blocks.0.mlp' in k]
    print(mlp_keys)  # Should show gate_proj, up_proj, down_proj

# Check if model uses them
model = load_model()
first_block = model.visual.blocks[0]
print(hasattr(first_block.mlp, 'gate_proj'))  # Should be True
```

##### Lesson 6: Statistical Analysis Beyond Values

**What We Did:**
```python
# Don't just compare values:
pt_val = [0.1, 0.2, -0.15]
mlx_val = [0.09, 0.21, -0.14]  # Close! Ship it?

# Compare RANGES (reveals scaling issues):
pt_range = pt_stats['max'] - pt_stats['min']   # 14.58
mlx_range = mlx_stats['max'] - mlx_stats['min']  # 0.51
ratio = pt_range / mlx_range  # 28.76x ← BIG PROBLEM!
```

**Why Ranges Matter:**
- Neural networks care about relative magnitudes
- Small absolute differences can hide massive scaling issues
- Ranges reveal amplification/attenuation in layers

**Real Impact:** Initial values looked "close enough" (mean difference ~0.01), but range analysis revealed the 28.8x MLP bug immediately.

#### Summary of Vision Tower Debugging

**Three Major Bugs Fixed:**

1. **VisionPatchEmbed Data Layout** (NCDHW vs NDHWC)
   - Impact: 10x improvement in exact matches
   - Fix: Local transpose, compute, transpose back

2. **VisionBlock Normalization** (LayerNorm vs RMSNorm)
   - Impact: Changed centering behavior
   - Fix: Use RMSNorm to match PyTorch

3. **VisionMLP Architecture** (Simple vs GLU-style)
   - Impact: 28.70x improvement in MLP outputs
   - Fix: Implement proper GLU with gate/up/down projections

**Debugging Time:** ~2 hours of systematic component isolation

**Key Success Factors:**
- ✅ Work backwards from known-good state
- ✅ Handle convention differences locally
- ✅ Isolate components within composite blocks
- ✅ Verify architecture against source
- ✅ Compare ranges, not just values
- ✅ Check both weight loading AND implementation

**Final Result:** Vision tower now matches PyTorch perfectly! MLP outputs within 0.03 of reference (14.55 vs 14.58).

### Session 10: The Importance of Comparing Apples to Apples (CRITICAL LESSON!)

**Task:** Debug 6.5x transformer output divergence after fixing vision tower

**What Happened:** Made a critical analysis error that led to 40 minutes of incorrect debugging.

#### The Mistake: Comparing Different Execution Stages

**Initial Finding (WRONG):**
```
After TXT_IN:
  PyTorch: range=35.25
  MLX:     range=315
  
Conclusion: txt_in is broken! 8.9x divergence!
```

**The Problem:** I was comparing:
- PyTorch line 622: State captured **AFTER txt_norm, BEFORE txt_in** → range=35.25
- MLX line 48: State captured **BEFORE txt_norm** → range=315

**These are completely different stages!**

#### The Debugger's Behavior

**CRITICAL:** Debugger breakpoints capture state **BEFORE the line executes**:

```python
# Line 48: hidden_states = self.txt_norm(encoder_hidden_states)
# Breakpoint at line 48 captures encoder_hidden_states BEFORE txt_norm runs
```

This means:
- Breakpoint at line 48 = input to that line
- To see output, you need breakpoint at line 49 or later

#### The Correct Analysis

When comparing the **actual same stages**:

```
First Block Input (after txt_norm + txt_in):
  PyTorch: range=1094
  MLX:     range=1094.7
  
Result: PERFECT MATCH! ✅
```

**txt_in was working correctly all along!**

#### What Was Really Wrong

The actual issue: **Loading PyTorch embeddings from disk caused numerical explosion in BOTH implementations**:
- PyTorch: Ranges grew to 24M by block 23
- MLX: Ranges grew to 239M by block 23 (9.7x worse)

The 6.5x final divergence wasn't due to txt_in being broken - it was due to accumulated numerical instability from incompatible embeddings.

#### Key Lessons

##### Lesson 1: Understand Your Tool

**Before comparing values, verify:**
1. What does the breakpoint capture? (before or after line execution?)
2. Are both traces at equivalent stages?
3. What transformations happened between the two points?

**Common pitfall:**
```python
# PyTorch
encoder_hidden_states = self.txt_norm(encoder_hidden_states)  # Line 621
encoder_hidden_states = self.txt_in(encoder_hidden_states)     # Line 622

# Breakpoint at 622 captures BEFORE txt_in!
```

##### Lesson 2: Verify Assumptions Early

When you find a surprising divergence:
1. **STOP** and verify you're comparing the same thing
2. Check the line numbers and what they actually capture
3. Trace through the execution order
4. Look at shapes to confirm transformations occurred

**In this case:**
- txt_in should change shape: 3584 → 3072
- Quick shape check would have revealed the comparison was invalid
- PyTorch at "line 622" was still [1, 1385, 3584]
- This means txt_in **hadn't executed yet**!

##### Lesson 3: Think Twice Before Drawing Conclusions

**Red flags that should trigger re-verification:**
- ❌ "This core component is completely broken" (unlikely if basic tests pass)
- ❌ "8.9x divergence in a simple linear layer" (very suspicious)
- ❌ "The architecture must be fundamentally wrong" (check assumptions first)

**Better approach:**
- ✅ "This divergence is larger than expected - let me verify my comparison is valid"
- ✅ "Before investigating txt_in, let me confirm both traces captured the same stage"
- ✅ "Let me check if dimensions changed as expected"

##### Lesson 4: Use Shape as Sanity Check

**Shapes are your friend:**
```python
# If txt_in is Linear(3584, 3072), output MUST be 3072
# If you see 3584, txt_in hasn't executed yet!

PT_shape = [1, 1385, 3584]  # ← txt_in didn't run!
MLX_shape = [1, 1385, 3584]  # ← txt_norm didn't run!

# These aren't comparable!
```

#### How to Avoid This

**1. Document your breakpoints clearly:**
```bash
# ✅ GOOD
"after_txt_in_line50"   # Clear: captures after line 49 executes

# ❌ BAD  
"txt_in_line49"         # Ambiguous: before or after?
```

**2. Always check shapes first:**
```bash
# Before analyzing values, verify shapes transformed as expected
jq '.variables.encoder_hidden_states.shape'  # Should be [1, 1385, 3072] after txt_in
```

**3. Compare execution flow explicitly:**
```python
# PyTorch execution order:
# Line 621: txt_norm runs
# Line 622: txt_in runs  ← Breakpoint captures BEFORE this
# Line 623: next operation

# MLX execution order:  
# Line 48: txt_norm runs ← Breakpoint captures BEFORE this
# Line 49: txt_in runs
```

**4. When in doubt, add intermediate breakpoints:**
```python
# Split compound operations for clarity:
normed = self.txt_norm(x)      # Breakpoint here
output = self.txt_in(normed)   # Breakpoint here
```

#### Time Cost of This Mistake

- **40 minutes** spent investigating txt_in when it was working correctly
- Created debug scripts, ran traces, analyzed weights
- Eventually discovered the comparison was invalid
- **Could have been avoided** with 30 seconds of shape verification

**ROI:** 30 seconds of verification > 40 minutes of misdirected debugging

#### Summary

**The Golden Rule:**
> "Before comparing two values, verify you're capturing the same execution stage. When in doubt, check shapes first."

**Signs you might be comparing wrong stages:**
- Shapes don't match expectations
- Divergence is surprisingly large for a simple operation
- One implementation "isn't working" despite passing other tests

**Always ask:**
1. What line is this breakpoint capturing?
2. Has the operation I'm interested in executed yet?
3. Are both traces at equivalent execution points?
4. Do shapes confirm the expected transformation occurred?

**Remember:** The debugger shows you state **before** line execution. Don't assume transformations have happened until you verify!

---

### Session 11: MLX Silent Out-of-Bounds Access (CRITICAL MLX BEHAVIOR!)

**Task:** Debug 123M numerical explosion in transformer blocks despite correct vision tower

**What Happened:** Time embeddings were 1.73x too large, causing a cascade that resulted in 123M explosion. The root cause was accessing an array out of bounds, but MLX didn't throw an error!

#### The Bug

```python
# BROKEN CODE (qwen_transformer.py, line 99)
time_step = config.scheduler.sigmas[t]  # t=1000, but sigmas only has 10 elements!
```

**What was happening:**
- `t` = 1000 (the timestep value)
- `config.scheduler.sigmas` = array of 10 sigma values (one per inference step)
- Accessing `sigmas[1000]` was **way out of bounds**
- **MLX didn't raise an error!** Instead returned garbage: `1.401298464324817e-45`

#### The Cascade of Errors

This tiny garbage value propagated through the entire model:

```
1. Time Embedding:    1.73x too large (244 vs 141)
                      ↓
2. Modulation Params: 2.24x too large (933 vs 416)
                      ↓
3. MLP Input:         2.6x too large (1348 vs 518)
                      ↓
4. MLP Output:        2.65x too large (299k vs 113k)
                      ↓
5. Block Output:      🔥 123M EXPLOSION! 🔥
```

#### How We Found It

**Systematic breakpoint tracing:**

```bash
# Set breakpoints BEFORE and AFTER time embedding computation
Breakpoint 1: line 628 (BEFORE time_embed call)
Breakpoint 2: line 631 (AFTER time_embed call)
Breakpoint 3: line 635 (at block loop)
```

**Critical findings:**
1. **PyTorch timestep input:** `[1.0]` (the sigma value)
2. **MLX timestep input:** `1.401e-45` (garbage!)
3. Traced back to `sigmas[1000]` access

#### The Root Cause: Index vs Value Confusion

The scheduler has two parallel arrays:
```python
scheduler.timesteps = [1000, 929, 851, 780, ...]  # Timestep VALUES
scheduler.sigmas    = [1.0, 0.929, 0.851, 0.780, ...]  # Corresponding sigma VALUES
```

**The confusion:**
- Loop passes: `t = 1000` (timestep VALUE)
- Code tried: `sigmas[t]` (using VALUE as INDEX)
- Should use: `sigmas[idx]` where `idx` is the step index (0-9)

#### The Fix

```python
# FIXED CODE
if isinstance(t, int):
    # Find which sigma corresponds to this timestep value
    timestep_idx = None
    for idx, ts in enumerate(config.scheduler.timesteps):
        if int(ts.item()) == t:
            timestep_idx = idx
            break
    if timestep_idx is None:
        raise ValueError(f"Timestep {t} not found in scheduler timesteps")
    time_step = config.scheduler.sigmas[timestep_idx]
else:
    time_step = t  # Already a sigma value
```

#### Results After Fix

**Time Embedding (Block 0 Entry):**
- Before: Range = 243.74, Mean = -0.0344
- After:  Range = 140.90, Mean = -0.0323
- PyTorch: Range = 141.0, Mean = -0.0321
- ✅ **PERFECT MATCH!**

**MLP Output:**
- Before: 299,469 range (2.65x PyTorch)
- After:  113,682 range (1.00x PyTorch)
- ✅ **PERFECT MATCH!**

**123M explosion:** ✅ **RESOLVED!**

#### Key Lessons

##### Lesson 1: MLX Doesn't Always Raise Errors for Out-of-Bounds Access

**Critical MLX behavior:**
```python
import mlx.core as mx

arr = mx.array([1.0, 2.0, 3.0])
print(arr[0])    # 1.0 ✅
print(arr[100])  # Garbage! (e.g., 1.401e-45) ❌ NO ERROR!
```

Unlike NumPy/PyTorch which raise `IndexError`, **MLX can silently return garbage values**.

**Why this is dangerous:**
- No immediate crash to alert you
- Garbage propagates through computation
- Results appear "reasonable" but are completely wrong
- Can cause cascading numerical explosions downstream

##### Lesson 2: Small Errors Can Cascade Dramatically

A tiny input error (`1e-45` vs `1.0`) caused:
- 1.73x error in embeddings
- 2.24x error in modulation
- 2.65x error in MLP
- **123M explosion in output!**

**In ML models with many layers:**
- Small errors multiply through the network
- Each layer amplifies the divergence
- Final outputs can be completely wrong

##### Lesson 3: Systematic Bracketing Finds Root Causes

**The debugging strategy that worked:**

```bash
# 1. Identify divergence point (transformer blocks)
# 2. Bracket the problem:
#    - Breakpoint BEFORE suspected operation
#    - Breakpoint AFTER suspected operation  
#    - Breakpoint at next stage
# 3. Compare inputs and outputs
# 4. Find where divergence starts
# 5. Work backwards to root cause
```

**In this case:**
```
Block output: 123M explosion    ← Problem identified
      ↑
MLP output: 2.65x too large     ← Still wrong
      ↑
MLP input: 2.6x too large       ← Still wrong
      ↑
Modulation: 2.24x too large     ← Still wrong
      ↑
Time embedding: 1.73x too large ← First divergence!
      ↑
Timestep input: 1e-45 vs 1.0    ← ROOT CAUSE! 🎯
```

##### Lesson 4: Check Your Indices!

**Common index/value confusion patterns:**

```python
# ❌ BAD - Using value as index
for timestep in timesteps:  # timestep = 1000, 929, 851...
    sigma = sigmas[timestep]  # Wrong! timestep is not an index

# ✅ GOOD - Using index
for idx, timestep in enumerate(timesteps):
    sigma = sigmas[idx]  # Correct! idx is 0, 1, 2...

# ✅ GOOD - Lookup by value
for timestep in timesteps:
    idx = timesteps.index(timestep)
    sigma = sigmas[idx]
```

##### Lesson 5: Add Assertions for MLX Code

**Defensive programming for MLX:**

```python
# ✅ Add bounds checking
assert 0 <= idx < len(sigmas), f"Index {idx} out of bounds for sigmas array of length {len(sigmas)}"
time_step = sigmas[idx]

# ✅ Add value validation  
time_step = sigmas[idx]
assert time_step > 1e-10, f"Suspiciously small timestep: {time_step} (possible out-of-bounds?)"

# ✅ Add shape validation
assert len(timesteps) == len(sigmas), "Timesteps and sigmas must have same length"
```

#### How to Spot This Issue

**Red flags that suggest out-of-bounds access:**

1. **Tiny garbage values:** `1e-45`, `1e-308`, etc.
2. **Cascading divergence:** Small input error → huge output error
3. **No error raised:** Code runs without crashes
4. **Index looks suspicious:** Using a large value (1000) with a small array (10 elements)

**Debugging checklist when you see tiny garbage values:**

```bash
# 1. Check array length
print(f"Array length: {len(sigmas)}")  # 10

# 2. Check index value  
print(f"Index: {t}")  # 1000 ← RED FLAG!

# 3. Verify bounds
print(f"Valid indices: 0 to {len(sigmas)-1}")  # 0 to 9

# 4. Add bounds check
if t >= len(sigmas):
    raise ValueError(f"Index {t} out of bounds for array of length {len(sigmas)}")
```

#### Time Cost vs Impact

- **Time spent finding bug:** 2 hours of systematic tracing
- **Impact of bug:** Model completely broken (123M explosion)
- **Fix complexity:** 15 lines of code
- **Result:** Model outputs now match PyTorch perfectly

**This was time well spent!** The systematic bracketing approach (breakpoints before/after operations) was essential for finding the root cause.

#### Summary

**The Golden Rule for MLX:**
> "MLX can silently return garbage for out-of-bounds access. Always validate array indices, especially when using values as indices."

**Key takeaways:**
1. ✅ MLX doesn't always raise errors for out-of-bounds access
2. ✅ Tiny garbage values can cascade into massive errors
3. ✅ Systematic bracketing (before/after breakpoints) finds root causes
4. ✅ Watch for index/value confusion (timestep VALUE vs step INDEX)
5. ✅ Add assertions for bounds checking in MLX code

**When you see:**
- Tiny garbage values (1e-45, 1e-308)
- Cascading numerical explosions
- Large index values with small arrays

**Ask:**
- Am I using a value as an index?
- Is this array access in bounds?
- Should I add assertions to catch this?

---

## Tips for AI Agents

### 🎯 Complete the Interactive Tutorial First

The tutorial teaches all essential debugging commands. The tips below cover advanced scenarios.

### Essential Best Practices ✅

**Advanced Breakpoints:**
- ✅ **Use absolute paths for line-based breakpoints** - e.g., `/Users/you/Desktop/mflux/src/mflux/models/flux/flux.py`
- ✅ **Test with simple breakpoints first** - Set a breakpoint in your script entry point to verify it works
- ✅ **Always install libraries in editable mode** - `uv pip install -e ~/Desktop/diffusers` (required for breakpoints in libraries)
- ✅ **Verify which version is loaded** - Use `eval "import diffusers; diffusers.__file__"` to check

**Model Comparison:**
- ✅ **Start with PyTorch reference** - Debug the reference implementation first
- ✅ **Use minimal inline changes** - Save/load data rather than writing comparison scripts
- ✅ **Comment out temporary debugging lines** - After using `sys.exit()` for saving data
- ✅ **Check trace files for offline analysis** - Saved in `mflux_debugger/traces/`

---

## Architecture

```
┌─────────────────────────────────────┐
│  Transport Layer (choose one)       │
│  - FastAPI REST API (recommended)   │
│  - MCP Server (Cursor integration)  │
└──────────────┬──────────────────────┘
               │
┌──────────────┴──────────────────────┐
│  DebuggerService (business logic)   │
│  - Session management                │
│  - Breakpoint management             │
│  - Variable formatting               │
│  - Rich context capture              │
│  - Async execution                   │
└──────────────┬──────────────────────┘
               │
┌──────────────┴──────────────────────┐
│  Debugger (core)                    │
│  - Selective tracing (user code)    │
│  - Library skipping (torch, mlx)    │
│  - Thread-safe state caching        │
│  - MLX auto-evaluation               │
└──────────────┬──────────────────────┘
               │
┌──────────────┴──────────────────────┐
│  TraceRecorder (persistence)        │
│  - Incremental auto-save             │
│  - Emergency save on crash           │
│  - Rich context capture              │
└─────────────────────────────────────┘
```

---

## Performance

**Overhead comparison:**

| Operation | Debugpy | Lightweight | Speedup |
|-----------|---------|-------------|---------|
| Startup | 2-3s | ~0.1s | 20-30x |
| Model loading | 60s+ (often hangs) | Normal speed | ∞ |
| Step over | ~0.5s | ~0.01s | 50x |
| Variable inspection | ~0.2s | ~0.001s | 200x |

**Key advantage:** Doesn't slow down model loading because it skips ML library internals!

---

## Requirements

```bash
# Core
uv pip install fastapi uvicorn

# For PyTorch debugging (reference)
uv pip install torch diffusers

# For MLX debugging
# (MLX is pre-installed on mflux project)
```

---

## Breakpointing in Library Code (Transformers, Diffusers, etc.)

**Good news:** The debugger fully supports breakpoints in library code (site-packages, editable installs, etc.)!

### ⚠️ Critical Rule: ALWAYS Verify You're Debugging the Right Implementation

**If breakpoints don't hit, DON'T assume the debugger is broken - you might be pointing to the wrong file!**

#### Common Mistakes:

1. **Assuming which library is used** without verification
2. **Multiple implementations exist** (transformers vs diffusers vs custom)
3. **Editable installs override site-packages** (~/Desktop/diffusers vs .venv/lib/.../diffusers)
4. **Model architecture versions differ** (Qwen2VL vs Qwen2_5_VL)

#### The Golden Rule: **Verify First, Debug Second**

**Example from real debugging session:**

```bash
# ❌ WRONG ASSUMPTION: "Diffusers uses transformers.Qwen2_5_VL"
# Set breakpoint at:
/Users/you/.venv/lib/python3.11/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:135

# Result: Breakpoint never hits! 😞

# ✅ CORRECT AFTER VERIFICATION: Diffusers has its OWN implementation
# The actual code path was:
/Users/you/Desktop/diffusers/src/diffusers/models/transformers/transformer_qwenimage.py:618

# Result: Breakpoint hits immediately! 🎉
```

**Real-world lesson:**
- We initially thought HF used `LayerNorm` in `PatchMerger`
- Set breakpoint in `transformers/.../ modeling_qwen2_vl.py` → never hit
- Checked logs → found diffusers uses its own code
- Set breakpoint in correct file → hit immediately
- **Discovered HF actually uses `Qwen2RMSNorm`, not LayerNorm!**

**Without runtime verification via breakpoints, we would have made the wrong fix.**

### How to Find the Right File

#### Method 1: Check Debug Logs (Recommended!)

The debugger prints which library files it encounters:

```bash
# Check logs while script runs
tail -f /tmp/debugger.log | grep "ℹ️  Library file"
```

You'll see output like:
```
ℹ️  Library file without breakpoint: /Users/you/Desktop/diffusers/src/diffusers/models/transformers/transformer_qwenimage.py
ℹ️  Library file without breakpoint: /Users/you/.venv/lib/python3.11/site-packages/transformers/modeling_utils.py
✅ Tracing library file with breakpoint: /Users/you/Desktop/diffusers/src/diffusers/models/transformers/transformer_qwenimage.py
```

**This shows you which files are ACTUALLY being executed!**

#### Method 2: Check Import Statements

```python
# In your script
from diffusers import DiffusionPipeline

# This might use diffusers' own Qwen implementation,
# NOT the transformers library!
```

#### Method 3: Check Library Installation Type

```bash
# Is it an editable install?
pip show diffusers | grep Location
# Location: /Users/you/Desktop/diffusers/src  ← Editable install!

pip show transformers | grep Location
# Location: /Users/you/.venv/lib/python3.11/site-packages  ← Standard install
```

### Installation Types and Paths

| Install Type | Location | Example Path |
|--------------|----------|--------------|
| **Standard pip install** | `site-packages` | `/.venv/lib/python3.11/site-packages/transformers/...` |
| **Editable install** (`-e .`) | Source directory | `/Users/you/Desktop/diffusers/src/diffusers/...` |
| **Local development** | Working directory | `/Users/you/projects/mylib/src/...` |

**The debugger traces ALL of these!** Just set your breakpoint in the correct file.

### Step-by-Step: Breakpoint in Library Code

**Example: Debug HuggingFace Diffusers model**

1. **Start debug session:**
```bash
mflux-debug-pytorch start debug_diffusers_edit.py
```

2. **Set breakpoint in library file:**
```bash
# Use absolute path to the library file
mflux-debug-pytorch break /Users/you/Desktop/diffusers/src/diffusers/models/transformers/transformer_qwenimage.py 618
```

3. **Run with automatic polling (library code takes time to reach):**
```bash
# Automatic polling with 120 second timeout for model loading
mflux-debug-pytorch continue --max-wait 120
```

4. **Inspect variables in library code:**
```bash
# Same as debugging your own code!
mflux-debug-pytorch eval "hidden_states.shape"
```

### Tips for Library Breakpoints

**✅ DO:**
- Use absolute paths to library files
- Check logs to verify which files are actually executed
- Test with a simple breakpoint first (like at the start of a forward() method)
- Use async execution for library code (model loading takes time)

**❌ DON'T:**
- Assume a library uses another library's implementation (e.g., diffusers may have its own Qwen code)
- Set breakpoints in files that aren't actually imported
- Expect instant breakpoint hits (library code is reached after model loading)

### Troubleshooting Library Breakpoints

**Breakpoint not hitting?**

1. **Check if the file is actually being executed:**
```bash
tail -f /tmp/debugger.log | grep "your_library_file.py"
```

If you DON'T see it in the logs, that file isn't being used!

2. **Look for the actual implementation:**
```bash
# See all library files being encountered
grep "ℹ️  Library file" /tmp/debugger.log | grep "your_model_name" | sort -u
```

3. **Verify the path matches exactly:**
```bash
# The debugger normalizes paths with resolve()
python3 -c "from pathlib import Path; print(Path('/path/to/file.py').resolve())"
```

4. **Check for multiple implementations:**
- Editable install in `/Desktop/diffusers/` (custom fork)
- Standard install in `site-packages/diffusers/` (PyPI version)
- Which one is imported first? Check `sys.path` order!

### Real-World Example: Debugging Qwen Vision Tower

**Goal:** Debug PatchMerger in HuggingFace code

**Challenge:** Diffusers has its own Qwen implementation!

**Solution:**
```bash
# ❌ Initially tried (wrong file):
transformers/models/qwen2_vl/modeling_qwen2_vl.py:262

# ✅ Checked logs, found actual file:
diffusers/models/transformers/transformer_qwenimage.py:618

# Set breakpoint in correct file → SUCCESS! ✅
```

**Key learning:** Always check the logs to see which files are ACTUALLY being executed.

---

## Troubleshooting

### Breakpoint not hitting?

**Check:**
1. **Is the library installed in editable mode?** This is the #1 cause!
   ```bash
   uv pip list | grep library_name
   # Should show a path like: /Users/you/Desktop/library_name
   ```
   If not, install in editable mode:
   ```bash
   uv pip install -e ~/Desktop/library_name
   ```
   Then restart the debug server.

2. **Is the path absolute?** Use absolute paths for `file_path`
   ```bash
   # ✅ Good
   /Users/you/Desktop/diffusers/src/diffusers/pipelines/flux/pipeline_flux.py
   
   # ❌ Bad
   ../diffusers/src/diffusers/pipelines/flux/pipeline_flux.py
   ```

3. **Is Python loading the right version?**
   ```bash
   mflux-debug-pytorch eval "import diffusers; diffusers.__file__"
   # Should show your editable install path
   ```

4. **Is the script running?** Check `/debug/status`

5. **Is the line executable?** Set breakpoint on actual code, not comments/blank lines

6. **Did the script crash?** If status shows "finished" but no breakpoints hit, check logs:
   ```bash
   # Check debugger server logs (if running in background)
   tail -50 /tmp/debugger.log
   
   # Or check the trace file for errors
   cat mflux_debugger/traces/your_session_*.json | grep -i error
   ```
   Common causes: Missing weights, import errors, KeyError in weight loading, incompatible dependencies

7. **Clean slate approach:** If breakpoints mysteriously don't hit, try this workflow:
   ```bash
   # 1. Terminate any existing debug session (automatic cleanup)
   mflux-debug-pytorch terminate
   
   # 2. Start new session (auto-starts server and cleans state)
   mflux-debug-pytorch start your_script.py
   
   # 3. Test with a SIMPLE breakpoint first (entry point of your script)
   mflux-debug-pytorch break /absolute/path/to/your_script.py 6
   
   # 4. Run and verify it hits (automatic polling)
   mflux-debug-pytorch continue --max-wait 10
   
   # 5. Check location
   mflux-debug-pytorch location
   
   # 6. If that works, clear it and set your real breakpoints
   mflux-debug-pytorch clear
   # Now set breakpoints in library code...
   ```

### Best Practice: Test Simple First, Then Complex

**❌ DON'T start with:**
```bash
# Immediately setting breakpoints deep in library code
mflux-debug-pytorch break /Users/you/Desktop/diffusers/.../pipeline.py 856
mflux-debug-pytorch continue
# 😱 Doesn't hit, now what's wrong?
```

**✅ DO start with:**
```bash
# First: Set breakpoint at entry point of YOUR script
mflux-debug-pytorch break /path/to/your_script.py 6
mflux-debug-pytorch continue
# ✅ Hits? Great! Debugger is working. Clear and set real breakpoints.
# ❌ Doesn't hit? Fix the fundamentals (editable install, server restart, etc)
```

**Why this matters:**
- ✅ Verifies debugger trace is working
- ✅ Confirms paths are resolving correctly
- ✅ Ensures clean session state
- ✅ Builds confidence before debugging complex library code

**Rule of thumb:** If unsure, start with breakpoints in your own script, then work your way into libraries.

### "Still running" after 60 seconds?

**This is normal for model loading!** The CLI handles this automatically with extended timeouts:
```bash
# Automatic polling with 60 second timeout (or adjust --max-wait as needed)
mflux-debug-mlx continue --max-wait 60

# For very slow models, increase the timeout
mflux-debug-mlx continue --max-wait 120
```

### Variables show `null`?

**For MLX:** Breakpoint might be before evaluation. Move to after `mx.eval()` call.

### Breakpoint never hit? Script might be crashing!

**Critical lesson:** If your breakpoint isn't being hit and everything looks correct, the script might be crashing BEFORE reaching the breakpoint.

**Real example from Qwen debugging:**
```bash
# Breakpoint set at line 193 ✅
# File is being traced ✅  
# Path matches ✅
# But breakpoint never hits ❌

# The problem: Script crashed at line 95!
AttributeError: 'LinearScheduler' object has no attribute 'timesteps'
```

**How to diagnose:**

1. **Run the script directly** (without debugger):
```bash
uv run python your_script.py
```

2. **Check if it completes successfully** - Does it crash? Where?

3. **Common silent failures:**
   - Missing attributes (wrong object type)
   - Import errors in nested modules
   - Configuration not passed through correctly
   - Early return/exception before breakpoint

4. **Add strategic print statements** BEFORE your breakpoint:
```python
print(f"🔍 DEBUG: Reached line 90", flush=True)  # Before suspected crash
print(f"🔍 DEBUG: About to access attribute", flush=True)
# Your code that might crash
print(f"✅ DEBUG: Made it past line 95", flush=True)
```

**Rule of thumb:** If debugger says the file is being traced but breakpoint never hits, run the script standalone first to verify it reaches that line.

### Lost trace after crash?

**Good news:** Traces are auto-saved after every step! Check:
```bash
ls -lt mflux_debugger/traces/*.json | head -1
```

---

## Credits

Developed through live debugging sessions comparing FLUX.1 and Qwen-Image PyTorch (Diffusers) and MLX implementations.

**Version 0.2.4** includes lessons learned from:
- **Library code breakpoint support** (transformers, diffusers, site-packages, editable installs)
- **Implementation detection** (finding which library code is actually executed)
- **Debug logging for library tracing** (ℹ️ markers show files being encountered)
- **Real-world model port verification** (QWEN-Image PyTorch→MLX)
- **Minimal code intervention philosophy** (no helper scripts)
- **Smart polling patterns** (progress indicators for async operations)
- **Editable installation discovery** (breakpoint path matching)
- **Clean slate debugging workflow** (test simple breakpoints first)
- **Silent crash detection** (breakpoint not hit → check if script crashes before reaching it)
- **Scheduler parameter preservation** (FlowMatchEulerDiscreteScheduler integration)
- Path normalization debugging
- Crash resilience testing
- Loop breakpoint management
- Async execution for ML workloads
- Tensor shape verification across frameworks
- Virtual environment best practices
- Stale server state troubleshooting

---

**Happy Debugging! 🐛 → ✅**
