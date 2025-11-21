# Weight Inspector API Documentation

## Current Interface (Simplified & Clean)

### CLI Commands

```bash
mflux-debug-inspect-weights <model_name> [options]
```

**Positional Arguments:**
- `model_name` - Model name (e.g., `Qwen/Qwen-Image`, `qwen-image`, `briaai/FIBO`)

**Options:**
- `--local-path PATH` - Load weights from local path instead of HuggingFace
- `--component COMPONENT` - Filter by component (e.g., `transformer_blocks`, `decoder`)
- `--search PATTERN` - Search for weights matching pattern
- `--weight PATH` - Inspect specific weight with full stats
- `--report` - Print full verification report (patterns, coverage, structure)
- `--format {hf,mlx}` - Weight format (default: `hf`)

**Modes (mutually exclusive):**
1. **Default** - Summary + Structure (always shows structure - most useful)
2. `--report` - Full verification report (patterns, coverage, verification, structure)
3. `--weight PATH` - Inspect specific weight
4. `--search PATTERN` - Search and list matching weights

**Design Principles:**
- ✅ Always show structure by default (no hidden information)
- ✅ Minimal API surface (removed redundant options)
- ✅ Visual output (shows actual types, shapes, sizes)
- ✅ Never lies (shows real data, not inferred)

### Python API

```python
from mflux_debugger.weight_inspector import WeightInspector

inspector = WeightInspector(raw_weights, mapped_weights, model_name)

# Inspection methods
inspector.list_all(format="hf")  # List all weight paths
inspector.get_tensor(path, format="hf")  # Get tensor by path
inspector.get_stats(path, format="hf")  # Get tensor statistics
inspector.search(pattern, format="hf")  # Search weights
inspector.pretty_print(path, format="hf")  # Pretty print tensor

# Display methods
inspector.print_summary()  # Print summary
inspector.print_tree(max_depth=3, component=None)  # Print tree view

# Verification methods (NEW)
inspector.detect_patterns()  # Detect structural patterns
inspector.analyze_mapping_coverage(mapping_patterns=None)  # Analyze coverage
inspector.verify_structure(expected_structure=None)  # Verify structure
inspector.compare_raw_vs_mapped(sample_size=10)  # Compare raw vs mapped
inspector.print_mapping_report()  # Comprehensive report
```

## Current Issues & Improvements Needed

### Issue 1: Coverage Analysis Doesn't Use Actual Mapping

**Problem:** `analyze_mapping_coverage()` has a simplified pattern matcher that doesn't actually use the real mapping patterns from `WeightMapper._build_flat_mapping()`.

**Current behavior:**
- Just counts raw vs mapped weights
- Doesn't verify which weights SHOULD be mapped
- Can't tell if unmatched weights are intentional or missing

**Solution:** Need to pass actual mapping patterns or access the WeightMapper's flat mapping.

### Issue 2: Report Doesn't Show Actual Structure

**Problem:** `print_mapping_report()` shows counts but not the actual nested structure.

**Current output:**
```
Mapped weights: 105
Coverage: 95.5%
```

**Missing:** What does the structure actually look like? Is `decoder.up_blocks` a list? How many items?

**Solution:** Add structure visualization showing:
- Actual nested structure with types (dict/list/tensor)
- List lengths
- Key names at each level

### Issue 3: No Way to Verify Mapping Correctness

**Problem:** Can't verify that a specific HF weight maps to the correct MLX path.

**Missing:** 
- Given HF key `decoder.up_blocks.0.resnets.0.conv1.weight`, what MLX path should it map to?
- Does it actually map there?
- What transform was applied?

**Solution:** Add `verify_mapping(hf_key)` method that:
1. Shows expected MLX path from mapping
2. Shows actual MLX path (if found)
3. Shows transform applied
4. Shows if values match (after transform)

### Issue 4: Structure Verification Needs Expected Structure

**Problem:** `verify_structure()` requires `expected_structure` dict but we never provide it.

**Current:** Always passes `None`, so only does basic checks.

**Solution:** 
- Auto-generate expected structure from detected patterns
- Or allow passing mapping class to infer expected structure

## Proposed Improvements

### 1. Enhanced Coverage Analysis

```python
def analyze_mapping_coverage(
    self, 
    mapping_class: Optional[Type[WeightMapping]] = None,
    num_blocks: Optional[int] = None,
    num_layers: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Analyze mapping coverage using actual mapping patterns.
    
    If mapping_class provided, builds actual flat mapping and verifies
    which weights match vs don't match.
    """
```

### 2. Structure Visualization

```python
def print_structure(self, max_depth: int = 4, show_types: bool = True):
    """
    Print actual nested structure showing:
    - Types (dict/list/tensor)
    - List lengths
    - Key names
    - Sample paths
    
    Example:
    decoder/
      conv_in/ (dict)
        conv3d/ (dict)
          weight: array[48, 256, 3, 3, 3]
          bias: array[256]
      up_blocks/ (list[4])
        [0]/
          resnets/ (list[3])
            [0]/
              conv1/ (dict)
                conv3d/ (dict)
                  weight: array[...]
    """
```

### 3. Mapping Verification

```python
def verify_mapping(
    self, 
    hf_key: str,
    mapping_class: Optional[Type[WeightMapping]] = None,
) -> Dict[str, Any]:
    """
    Verify mapping for a specific HF key.
    
    Returns:
        {
            "hf_key": "...",
            "expected_mlx_path": "...",
            "actual_mlx_path": "...",  # or None if not found
            "transform": "transpose_conv3d_weight" or None,
            "matches": bool,
            "raw_tensor": array,
            "mapped_tensor": array or None,
            "values_match": bool or None,  # After transform
        }
    """
```

### 4. Enhanced Report

```python
def print_mapping_report(
    self,
    mapping_class: Optional[Type[WeightMapping]] = None,
    show_structure: bool = True,
    show_unmatched: bool = True,
    verify_samples: int = 5,
):
    """
    Enhanced report that:
    1. Shows detected patterns
    2. Shows actual structure (if show_structure=True)
    3. Uses real mapping patterns for coverage (if mapping_class provided)
    4. Shows unmatched weights with reasons (if show_unmatched=True)
    5. Verifies sample mappings (if verify_samples > 0)
    """
```

## Usage Examples

### Usage Examples

```bash
# Default: Summary + Structure (always shows structure)
mflux-debug-inspect-weights briaai/FIBO

# Full verification report (everything)
mflux-debug-inspect-weights briaai/FIBO --report

# Filter by component (structure for that component only)
mflux-debug-inspect-weights briaai/FIBO --component decoder

# Search for weights
mflux-debug-inspect-weights briaai/FIBO --search "decoder.up_blocks.0"

# Inspect specific weight
mflux-debug-inspect-weights briaai/FIBO --weight "decoder.conv_in.weight"
```

**What you always see:**
- Summary: Counts, components
- Structure: Actual nested structure with types, list lengths, array shapes
- No hidden information - everything is visible

### Proposed Enhanced Usage

```bash
# Report with actual structure
mflux-debug-inspect-weights briaai/FIBO --report --show-structure

# Verify specific mapping
mflux-debug-inspect-weights briaai/FIBO --verify-mapping "decoder.up_blocks.0.resnets.0.conv1.weight"

# Coverage with actual mapping patterns
mflux-debug-inspect-weights briaai/FIBO --coverage --mapping-class FIBOWeightMapping
```

## Design Principles

1. **Never Lie** - Always show actual data, not inferred summaries
2. **Show Structure** - Don't just count, show what the structure actually is
3. **Verify Mappings** - Can verify specific mappings are correct
4. **Explain Gaps** - If weights are unmatched, explain why (optional, attention, etc.)
5. **Use Real Data** - Use actual mapping patterns, not simplified matchers

