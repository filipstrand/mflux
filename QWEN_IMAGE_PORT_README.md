## Qwen Image Port: Hard‑Won Truths and Techniques

This document consolidates the essential, reusable lessons from porting Qwen Image components to MLX. It omits block-by-block stories, numeric journeys, and transient scripts. Use this as the playbook for debugging and completing the remaining components.

### Component Status Overview

#### ✅ **COMPLETED COMPONENTS**
- **VAE Decoder**: ✅ **COMPLETED** - Successfully implemented using **debugger-driven manual weight mapping**
- **Main Image Transformer**: ✅ Working - processes latents correctly, all layers functional
- **Scheduler**: ✅ Working - timestep and noise scheduling matches reference
- **Text Encoder Internal Transformer**: ✅ Working - QKV linear projections and transformer layers functioning correctly

#### 🔧 **IN PROGRESS**
- **VAE Encoder**: 🔧 **NEXT UP** - Apply same debugger-driven manual mapping approach that worked for decoder
  
  **CRITICAL CLARIFICATION**: The Qwen model has **TWO separate transformers**:
  1. **Text Encoder Transformer** (✅ WORKING): Internal transformer within the text encoder that processes text tokens into embeddings. This is a standard language model transformer (28 layers, 3584 hidden size).
  2. **Main Image Transformer** (✅ WORKING): The main diffusion transformer that processes image latents. This handles the actual image generation and is already working correctly.

### Core principles
- **Deterministic first**: fixed seeds, identical inputs, identical code paths
- **Find the first divergence**: layer-by-layer comparison before any refactors
- **Architecture over precision**: confirm structure and dataflow before blaming dtype
- **Ground truth = raw weights**: inspect the pretrained file's keys; do not trust assumptions
- **🎯 CRITICAL: Component boundaries vs computation sequence**: When porting from reference implementations (e.g., diffusers), focus on preserving the LINEAR SEQUENCE of operations, not where they're grouped. A computation might be inside the VAE in diffusers but outside in MLX (or vice versa) - this is just convention. What matters is that ALL steps are performed in the correct ORDER without duplication or omission. Naive line-by-line porting can lead to double computation if the same operation exists in multiple places.

### Component boundaries and computation sequence
**CRITICAL INSIGHT**: Component organization is convention, operation sequence is law.

#### The Problem:
- Diffusers might put normalization inside the VAE, while MLX puts it outside
- Different implementations group operations differently (pre-processing, post-processing, etc.)
- Naive porting can lead to duplicated or missing operations

#### The Solution:
1. **Map the complete computation flow**: Trace the full path from raw input to final output in the reference
2. **Identify ALL operations**: List every computation step regardless of which component contains it
3. **Preserve sequence, not grouping**: Ensure your MLX implementation performs the same operations in the same order
4. **Check for duplication**: Verify no operation is performed twice (once in component, once outside)
5. **Check for omission**: Verify no operation is skipped because "it should be handled elsewhere"

#### Example Scenarios:
- **Normalization**: Reference has `latents = latents / std + mean` inside VAE, but MLX has it in pipeline
- **Scaling**: Reference applies `latents *= 0.18215` in encoder, but MLX applies it in pipeline  
- **Preprocessing**: Reference does image normalization in processor, MLX does it in VAE

#### Debugging Strategy:
- Add strategic prints at component boundaries to see what values are being passed
- Compare intermediate tensors at the SAME LOGICAL POINTS, not the same component boundaries
- Focus on end-to-end correctness rather than component-by-component matching

#### **🎯 CRITICAL: Expand scope beyond the target component**
When asked to port a specific component (e.g., "port the VAE encoder"), **DO NOT** work in isolation:

1. **Understand the broader integration**: How does this component fit into the complete pipeline?
2. **Trace input sources**: What preprocessing happens before your component? Where does the input come from?
3. **Trace output destinations**: Where does your component's output go? What postprocessing happens after?
4. **Identify integration points**: What assumptions does your component make about input format, scaling, normalization?
5. **Map cross-component operations**: Are there operations that span multiple components or could be duplicated?

**Example**: When porting "VAE encoder," also examine:
- Image preprocessing pipeline (normalization, resizing, format conversion)
- How latents are passed to the main transformer
- Whether scaling/normalization happens in VAE vs pipeline vs main transformer
- Integration with scheduler and noise handling

**Why this matters**: Tunnel vision on a single component often misses critical integration details that cause subtle bugs in the complete system.

### Reference-side workflow (PyTorch)
- Run the diffusers reference in a separate repo with fixed seed and fixed resolution
- **Modify production code, not test scripts**: Add prints/breakpoints directly in the real forward pass rather than creating separate debug scripts that may contain bugs
- Export named tensors for MLX ingestion; use clear, stable names (e.g., `vae/pre_decode_latents.pt`, `vae/up_block_1/res_0_out.pt`)
- Keep all run conditions repeatable; store the exact prompt/settings alongside tensors
- **Trace active code paths**: Add strategic prints to see which branches actually execute (dropout, conditional blocks, etc.)

### Advanced Debugging: Reference Implementation Instrumentation

**When standard approaches fail**, instrument the reference implementation for granular tensor comparison:

#### Technique: Monkey-patch Reference for Detailed Tensor Capture
1. **Identify target layer**: Focus on the specific layer/component showing divergence
2. **Monkey-patch forward methods**: Replace original methods with instrumented versions that save intermediate tensors
3. **Granular tensor saving**: Save every intermediate computation with descriptive names:
   ```python
   torch.save(query_states.cpu().to(torch.float32), f"{debug_dir}/layer0_02_query_states.pt")
   torch.save(key_states.cpu().to(torch.float32), f"{debug_dir}/layer0_03_key_states.pt")
   torch.save(attn_weights.cpu().to(torch.float32), f"{debug_dir}/layer0_16_attn_weights.pt")
   ```
4. **Run reference once**: Generate complete tensor trace from single reference run
5. **Step-by-step MLX comparison**: Compare MLX implementation against each saved tensor

#### Benefits of This Approach:
- **Pinpoint precision**: Identify exact computation where divergence occurs
- **No hypothesis needed**: Let the data show where the problem is
- **Efficient debugging**: Run expensive reference once, compare MLX implementation multiple times
- **Avoid guesswork**: Replace assumptions with actual numerical evidence

#### When to Use:
- Standard layer-by-layer comparison shows divergence but root cause unclear
- Need to isolate specific operation causing differences
- Multiple components interact making it hard to identify the source

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

### 🌟 **PRIMARY STRATEGY: Debugger-Driven Manual Weight Mapping**

**This is the PREFERRED approach for all weight assignment. Avoid complex automated mapping systems.**

#### **The Flux-Style Approach (Recommended)**
1. **Print ALL weight keys from diffusers**: Add debug output to see every single weight key and shape from the pretrained model
   ```python
   print("🔍 ALL decoder weight keys from diffusers:")
   decoder_keys = sorted([k for k in weights.keys() if k.startswith("decoder.")])
   for i, key in enumerate(decoder_keys):
       print(f"  {i+1:2d}. {key}: {weights[key].shape}")
   ```

2. **Inspect MLX model structure by eye**: Read your MLX model code to understand the exact nested structure (lists, dicts, parameter names)

3. **Write explicit manual mappings**: Create simple, direct mappings like Flux does:
   ```python
   # Simple direct mappings (like Flux)
   weights["decoder"]["conv_in"] = {"conv3d": {
       "weight": diffusers_weights["decoder.conv_in.weight"],
       "bias": diffusers_weights["decoder.conv_in.bias"]
   }}
   
   # Manual structure building for nested components
   weights["decoder"]["mid_block"]["resnets"] = [{}, {}]
   for i in range(2):
       resnet = weights["decoder"]["mid_block"]["resnets"][i]
       resnet["conv1"] = {"conv3d": {
           "weight": diffusers_weights[f"decoder.mid_block.resnets.{i}.conv1.weight"],
           "bias": diffusers_weights[f"decoder.mid_block.resnets.{i}.conv1.bias"]
       }}
   ```

#### **Why This Approach Works**
- **Crystal clear mapping**: Every assignment is explicit and traceable
- **No hidden complexity**: No loops, automation, or clever algorithms that can fail silently
- **Easy debugging**: When something breaks, you can see exactly which line is responsible  
- **Matches model structure exactly**: You build the dictionary structure to match your MLX model precisely
- **Proven reliable**: Works identically to Flux's clean, simple approach

#### **Avoid These Anti-Patterns**
- ❌ **Complex automated mapping systems** with loops and clever logic
- ❌ **Generic tree-building algorithms** that try to be smart about structure
- ❌ **Helper functions with intricate logic** for mapping weight names
- ❌ **Assumptions about structure** - always inspect the actual keys and shapes

### Weight mapping essentials
- **🎯 USE THE DEBUGGER-DRIVEN APPROACH ABOVE**: This is the primary recommended strategy
- **Plot full hierarchy**: Print/inspect the complete weight dictionary structure from pretrained files; this must exactly match your MLX module hierarchy
- **Exact path matching**: A single level mismatch in the hierarchy will cause silent weight loading failures, leaving random weights
- **Lists vs dicts**: list modules need actual Python lists, not dicts of "0", "1", ...
- **No double nesting**: don't introduce containers (e.g., `conv3d`) twice in different layers
- **Verify weight assignment**: After loading, spot-check actual parameter values (first few numbers) against reference implementation to confirm weights were set correctly
- **Fast health checks**: shapes/dtypes; bias non‑zero; weight std within expected ranges; identical values across multiple runs (no randomness)
- **🎯 CRITICAL: Examine actual structure, never assume**: When debugging weight issues, add debug output to see the exact nested dictionary structure being created vs. what the MLX model expects. Don't make assumptions about parameter names (e.g., diffusers uses `gamma`/`beta`, MLX expects `weight`/`bias`)
- **🎯 CRITICAL: Transposes in computation, NOT in weight loading**: 
  - **Assign weights exactly as they exist in diffusers** - no transposes during loading
  - **Handle all transposes in the computation stage** (inside `__call__` methods)
  - **WHY this matters**: If you load weights without any transformations, you can easily write tests to verify that what you assigned is indeed what you expect by directly comparing against the original diffusers weights. If you transpose during loading, verification becomes much harder because you have to account for those transposes in your tests, adding complexity and potential for bugs.
  - **Debugging benefit**: When something breaks, you can immediately verify "did the weights load correctly?" by spot-checking raw values against diffusers, without having to mentally undo transposes

### Required layout conversions (PERFORM IN COMPUTATION, NOT WEIGHT LOADING)
- Conv2D: PyTorch `(out, in, h, w)` → MLX `(out, h, w, in)` via transpose `(0, 2, 3, 1)`
- Conv3D: PyTorch `(out, in, d, h, w)` → MLX `(out, d, h, w, in)` via transpose `(0, 2, 3, 4, 1)`

**Implementation pattern**: Transpose weights temporarily inside `__call__` methods:
```python
def __call__(self, x: mx.array) -> mx.array:
    # Transpose weight temporarily for MLX conv
    original_weight = self.conv3d.weight
    if len(original_weight.shape) == 5:  # 3D conv
        mlx_weight = mx.transpose(original_weight, (0, 2, 3, 4, 1))
        self.conv3d.weight = mlx_weight
        x = self.conv3d(x)
        # Restore original weight 
        self.conv3d.weight = original_weight
    return x
```

**Why this pattern**: Weights remain in original diffusers format for easy verification, transposes only occur during computation.

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
- **🌟 Use debugger-driven manual weight mapping (PRIMARY STRATEGY)**
- Print ALL diffusers weight keys and inspect MLX model structure by eye  
- Write explicit manual mappings - no complex automation
- **Assign weights exactly as they exist in diffusers** - no transposes during loading
- **Implement transposes in computation only** (inside `__call__` methods) - this makes weight verification much easier
- After loading, spot‑check: shapes, dtypes, means/stds, non‑zero biases (can directly compare to diffusers!)
- Hook critical boundaries; compare tensors with strict tolerances
- Fix the first divergence; re‑run; repeat
- Keep inputs and random states identical across frameworks

### Success criteria (per stage)
- Early layers (post‑quant, conv_in): ≤1e‑2 max diff, stable ranges
- Complex blocks: ≤1e‑3 to 1e‑4 where practical; stable downstream behavior
- End‑to‑end: visually equivalent images from identical latents/settings

### Current Phase: VAE Decoder COMPLETED ✅

**Status**: ✅ **COMPLETED** - Successfully implemented using debugger-driven manual weight mapping approach.

**🌟 BREAKTHROUGH: Debugger-Driven Manual Mapping**
The VAE decoder was successfully completed using the new **debugger-driven, manual Flux-style approach**:

1. **Printed all 106+ diffusers weight keys** to understand exact structure
2. **Inspected MLX model by eye** to understand nested lists/dicts structure  
3. **Wrote explicit manual mappings** - no automation, no complex logic
4. **Result**: Clean, working implementation in ~120 lines vs. 350+ lines of complex automation

**Key Success Factors**:
- **Crystal clear mappings**: Every assignment explicit and traceable
- **No hidden complexity**: Simple, direct mapping like Flux uses
- **Debuggable**: When issues arise, exact line responsible is obvious
- **Easily verifiable**: Weights loaded exactly as they exist in diffusers - can directly compare for verification
- **Testable weight loading**: No mental gymnastics to verify correct weight assignment
- **Reliable**: Matches MLX model structure perfectly

**Lessons for Future Components**:
- ❌ **Avoid complex automated mapping systems** - they're hard to debug and maintain
- ✅ **Use debugger-driven manual approach** - much cleaner and more reliable
- ✅ **Print all weight keys first** - understand the structure before coding
- ✅ **Manual structure building** - explicit is better than clever

### Next Phase: VAE Encoder  

**Status**: 🔧 **NEXT UP** - Apply the same debugger-driven manual mapping approach that worked for the decoder

### Notes for future phases
- Text encoder internal transformer is now completed and working
- Focus on remaining VAE encoder weight assignment and architecture issues
- Apply lessons learned about examining actual structures rather than making assumptions

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
