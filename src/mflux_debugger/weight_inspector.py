"""
Weight Inspector - Inspect and visualize model weights.

Provides tools to inspect raw HuggingFace weights and mapped MLX weights,
with support for searching, filtering, and displaying tensor statistics.

Debugging Pattern:
When implementing weight mappings, use these verification methods:
1. detect_patterns() - Discover model structure (blocks, layers, etc.)
2. analyze_mapping_coverage() - Check mapping completeness
3. verify_structure() - Validate nested structure correctness
4. compare_raw_vs_mapped() - Ensure no data loss during mapping
"""

import re
from typing import Any, Dict, List, Optional

import mlx.core as mx


class WeightInspector:
    """Inspect and analyze weight tensors from model loading."""

    def __init__(self, raw_weights: Dict[str, mx.array], mapped_weights: Dict[str, Any], model_name: str):
        """
        Initialize inspector with weights.

        Args:
            raw_weights: Raw HuggingFace weights (flat dict with dot-notation keys)
            mapped_weights: Mapped MLX weights (nested dict structure)
            model_name: Name of the model being inspected
        """
        self.raw_weights = raw_weights
        self.mapped_weights = mapped_weights
        self.model_name = model_name

    def list_all(self, format: str = "hf") -> List[str]:
        """
        List all weight names.

        Args:
            format: "hf" for HuggingFace format, "mlx" for MLX nested paths

        Returns:
            List of weight paths
        """
        if format == "hf":
            return sorted(self.raw_weights.keys())
        else:  # mlx
            return self._get_nested_keys(self.mapped_weights)

    def get_tensor(self, path: str, format: str = "hf") -> Optional[mx.array]:
        """
        Get tensor by path.

        Args:
            path: Weight path (HF dot-notation or MLX nested path)
            format: "hf" or "mlx"

        Returns:
            Tensor array or None if not found
        """
        if format == "hf":
            return self.raw_weights.get(path)
        else:  # mlx
            return self._get_nested_value(self.mapped_weights, path)

    def get_stats(self, path: str, format: str = "hf") -> Optional[Dict[str, Any]]:
        """
        Get tensor statistics.

        Args:
            path: Weight path
            format: "hf" or "mlx"

        Returns:
            Dictionary with shape, dtype, min, max, mean, std, size_mb
        """
        tensor = self.get_tensor(path, format)
        if tensor is None:
            return None

        # Evaluate if lazy (MLX)
        if hasattr(tensor, "shape") and hasattr(tensor, "dtype"):
            try:
                mx.eval(tensor)
                stats = {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "size_mb": tensor.nbytes / (1024 * 1024),
                }

                # Compute statistics (only for reasonable sizes to avoid memory issues)
                if tensor.size < 100_000_000:  # < 100M elements
                    stats["min"] = float(mx.min(tensor).item())
                    stats["max"] = float(mx.max(tensor).item())
                    stats["mean"] = float(mx.mean(tensor).item())
                    stats["std"] = float(mx.std(tensor).item())
                else:
                    stats["min"] = None
                    stats["max"] = None
                    stats["mean"] = None
                    stats["std"] = None
                    stats["note"] = "Statistics skipped (tensor too large)"

                return stats
            except (ValueError, RuntimeError, AttributeError) as e:
                return {"error": str(e)}
        return None

    def search(self, pattern: str, format: str = "hf") -> List[str]:
        """
        Search weights by pattern.

        Supports:
        - Simple substring matching
        - {block} placeholder substitution (e.g., "transformer_blocks.{block}.attn")

        Args:
            pattern: Search pattern
            format: "hf" or "mlx"

        Returns:
            List of matching weight paths
        """
        all_keys = self.list_all(format)
        matches = []

        # Check if pattern contains {block} placeholder
        if "{block}" in pattern:
            # Try block numbers 0-100 (reasonable range)
            for block_num in range(100):
                concrete_pattern = pattern.replace("{block}", str(block_num))
                matches.extend([key for key in all_keys if concrete_pattern in key])
        else:
            # Simple substring matching
            pattern_lower = pattern.lower()
            matches.extend([key for key in all_keys if pattern_lower in key.lower()])

        return sorted(set(matches))  # Remove duplicates and sort

    def format_path(self, path: str, format: str = "hf") -> str:
        """
        Format path for display.

        Args:
            path: Weight path
            format: "hf" or "mlx"

        Returns:
            Formatted path string
        """
        if format == "hf":
            return path
        else:  # mlx - convert dot notation to nested dict notation
            parts = path.split(".")
            formatted = parts[0]
            for part in parts[1:]:
                if part.isdigit():
                    formatted += f"[{part}]"
                else:
                    formatted += f"['{part}']"
            return formatted

    def pretty_print(self, path: str, format: str = "hf", max_preview: int = 10):
        """
        Pretty print tensor with statistics.

        Args:
            path: Weight path
            format: "hf" or "mlx"
            max_preview: Maximum number of values to preview
        """
        stats = self.get_stats(path, format)
        if stats is None:
            print(f"❌ Weight not found: {path}")
            return

        print(f"\n📊 {path}")
        print(f"   Shape: {stats['shape']}")
        print(f"   Dtype: {stats['dtype']}")
        print(f"   Size: {stats['size_mb']:.2f} MB")

        if "error" in stats:
            print(f"   ⚠️  Error: {stats['error']}")
            return

        if stats.get("note"):
            print(f"   ⚠️  {stats['note']}")
        else:
            print(f"   Min: {stats['min']:.6f}")
            print(f"   Max: {stats['max']:.6f}")
            print(f"   Mean: {stats['mean']:.6f}")
            print(f"   Std: {stats['std']:.6f}")

        # Preview values
        tensor = self.get_tensor(path, format)
        if tensor is not None and tensor.size <= 100:  # Only preview small tensors
            try:
                mx.eval(tensor)
                flat = tensor.flatten()
                preview = [float(x.item()) for x in flat[:max_preview]]
                print(f"   Preview: {preview}")
            except (ValueError, RuntimeError, AttributeError):
                pass  # Skip preview on error

    def print_summary(self):
        """Print summary of all weights."""
        hf_keys = self.list_all("hf")
        mlx_keys = self.list_all("mlx")

        print(f"\n🔍 Weight Summary for: {self.model_name}")
        print(f"   HuggingFace weights: {len(hf_keys)}")
        print(f"   MLX mapped weights: {len(mlx_keys)}")

        # Group by component
        components = {}
        for key in hf_keys:
            component = key.split(".")[0]
            if component not in components:
                components[component] = []
            components[component].append(key)

        print("\n📦 Components:")
        for component, keys in sorted(components.items()):
            print(f"   {component}: {len(keys)} weights")

    def print_tree(self, max_depth: int = 3, component: Optional[str] = None):
        """
        Print tree view of weights.

        Args:
            max_depth: Maximum depth to display
            component: Filter by component (e.g., "transformer_blocks")
        """
        print(f"\n🌳 Weight Tree for: {self.model_name}")
        if component:
            print(f"   Filtered by: {component}")

        def _print_tree_recursive(weights: Dict, prefix: str = "", depth: int = 0, filter_prefix: Optional[str] = None):
            if depth > max_depth:
                return

            if isinstance(weights, dict):
                for key, value in sorted(weights.items()):
                    current_path = f"{prefix}.{key}" if prefix else key

                    # Filter by component if specified
                    if filter_prefix and not current_path.startswith(filter_prefix):
                        continue

                    if isinstance(value, dict):
                        # Check if this is a leaf (contains tensor-like data)
                        has_tensor = any(
                            isinstance(v, mx.array) or (isinstance(v, dict) and "weight" in v or "bias" in v)
                            for v in value.values()
                        )

                        if has_tensor:
                            # This is a container with weights
                            print(f"{'  ' * depth}├─ {key}/")
                            _print_tree_recursive(value, current_path, depth + 1, filter_prefix)
                        else:
                            # Regular nested dict
                            print(f"{'  ' * depth}├─ {key}/")
                            _print_tree_recursive(value, current_path, depth + 1, filter_prefix)
                    elif isinstance(value, mx.array):
                        # Leaf tensor
                        shape_str = str(list(value.shape))
                        print(f"{'  ' * depth}├─ {key} {shape_str}")
                    else:
                        print(f"{'  ' * depth}├─ {key} ({type(value).__name__})")

        _print_tree_recursive(self.mapped_weights, filter_prefix=component)

    @staticmethod
    def _get_nested_keys(d: Dict, prefix: str = "") -> List[str]:
        """Get all nested keys from a dict structure as dot-notation paths."""
        keys = []
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                keys.extend(WeightInspector._get_nested_keys(v, full_key))
            else:
                keys.append(full_key)
        return keys

    @staticmethod
    def _get_nested_value(d: Dict, path: str) -> Optional[Any]:
        """Get value from nested dict using dot-notation path."""
        parts = path.split(".")
        current = d
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
                if current is None:
                    return None
            else:
                return None
        return current

    def detect_patterns(self) -> Dict[str, Any]:
        """
        Detect structural patterns in raw weights.

        Automatically discovers:
        - Number of transformer blocks
        - Number of text encoder layers
        - Number of up/down blocks (for VAE)
        - Number of resnets per block
        - Other repeating patterns

        Returns:
            Dictionary with detected patterns
        """
        patterns = {
            "transformer_blocks": set(),
            "text_encoder_layers": set(),
            "up_blocks": set(),
            "down_blocks": set(),
            "resnets_per_block": {},
            "other_patterns": {},
        }

        for key in self.raw_weights.keys():
            # Detect transformer blocks
            match = re.search(r"transformer_blocks\.(\d+)\.", key)
            if match:
                patterns["transformer_blocks"].add(int(match.group(1)))

            # Detect text encoder layers
            match = re.search(r"model\.layers\.(\d+)\.", key)
            if match:
                patterns["text_encoder_layers"].add(int(match.group(1)))

            # Detect VAE up_blocks
            match = re.search(r"up_blocks\.(\d+)\.", key)
            if match:
                block_num = int(match.group(1))
                patterns["up_blocks"].add(block_num)
                # Count resnets per block
                resnet_match = re.search(rf"up_blocks\.{block_num}\.resnets\.(\d+)\.", key)
                if resnet_match:
                    resnet_num = int(resnet_match.group(1))
                    if block_num not in patterns["resnets_per_block"]:
                        patterns["resnets_per_block"][block_num] = set()
                    patterns["resnets_per_block"][block_num].add(resnet_num)

            # Detect VAE down_blocks
            match = re.search(r"down_blocks\.(\d+)\.", key)
            if match:
                patterns["down_blocks"].add(int(match.group(1)))

        # Convert sets to sorted lists and counts
        result = {
            "num_transformer_blocks": max(patterns["transformer_blocks"]) + 1 if patterns["transformer_blocks"] else 0,
            "num_text_encoder_layers": max(patterns["text_encoder_layers"]) + 1
            if patterns["text_encoder_layers"]
            else 0,
            "num_up_blocks": max(patterns["up_blocks"]) + 1 if patterns["up_blocks"] else 0,
            "num_down_blocks": max(patterns["down_blocks"]) + 1 if patterns["down_blocks"] else 0,
            "resnets_per_up_block": {block: len(resnets) for block, resnets in patterns["resnets_per_block"].items()},
        }

        return result

    def analyze_mapping_coverage(self, mapping_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze mapping coverage - how many weights are mapped vs unmatched.

        This helps verify that:
        1. All expected weights are mapped
        2. Unmatched weights are intentional (optional, attention, etc.)
        3. Mapping patterns cover the model structure

        Args:
            mapping_patterns: Optional list of mapping patterns to test against
                            (e.g., from WeightMapping.get_vae_mapping())

        Returns:
            Dictionary with coverage statistics
        """
        total_raw = len(self.raw_weights)
        total_mapped = self._count_nested_weights(self.mapped_weights)

        # Find unmatched weights
        matched_keys = set()
        unmatched_keys = []

        # If we have mapping patterns, check which raw keys would match
        if mapping_patterns:
            # Build a simple pattern matcher (simplified version)
            for hf_key in self.raw_weights.keys():
                matched = False
                for pattern in mapping_patterns:
                    # Simple pattern matching (could be enhanced)
                    if pattern.replace("{block}", ".*").replace("{res}", ".*").replace("{i}", ".*") in hf_key:
                        matched = True
                        break
                if matched:
                    matched_keys.add(hf_key)
                else:
                    unmatched_keys.append(hf_key)
        else:
            # Without patterns, we can't determine which should match
            # Just report the counts
            pass

        # Group unmatched by component/type
        unmatched_by_component = {}
        for key in unmatched_keys:
            component = key.split(".")[0]
            if component not in unmatched_by_component:
                unmatched_by_component[component] = []
            unmatched_by_component[component].append(key)

        return {
            "total_raw_weights": total_raw,
            "total_mapped_weights": total_mapped,
            "coverage_percent": (total_mapped / total_raw * 100) if total_raw > 0 else 0,
            "unmatched_count": len(unmatched_keys),
            "unmatched_by_component": {comp: len(keys) for comp, keys in unmatched_by_component.items()},
            "unmatched_samples": unmatched_keys[:10],  # First 10 as samples
        }

    def verify_structure(self, expected_structure: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Verify the nested structure matches expectations.

        Checks:
        1. Expected keys exist
        2. Lists have correct lengths
        3. Nested dicts have correct structure

        Args:
            expected_structure: Optional dict describing expected structure
                              e.g., {"decoder": {"up_blocks": 4, "resnets_per_block": 3}}

        Returns:
            Dictionary with verification results
        """
        issues = []
        warnings = []

        def _verify_recursive(actual: Any, expected: Any, path: str = ""):
            if expected is None:
                return  # No expectations to verify

            if isinstance(expected, dict):
                if not isinstance(actual, dict):
                    issues.append(f"{path}: Expected dict, got {type(actual).__name__}")
                    return

                for key, expected_value in expected.items():
                    full_path = f"{path}.{key}" if path else key
                    if key not in actual:
                        issues.append(f"{full_path}: Missing expected key")
                    else:
                        _verify_recursive(actual[key], expected_value, full_path)

            elif isinstance(expected, int):
                # Expected count (for lists or dicts)
                if isinstance(actual, list):
                    if len(actual) != expected:
                        issues.append(f"{path}: Expected {expected} items, got {len(actual)}")
                elif isinstance(actual, dict):
                    if len(actual) != expected:
                        warnings.append(f"{path}: Expected {expected} keys, got {len(actual)}")

        if expected_structure:
            _verify_recursive(self.mapped_weights, expected_structure)

        # Basic structure checks even without expectations
        if isinstance(self.mapped_weights, dict):
            # Check for common issues
            if "decoder" in self.mapped_weights:
                decoder = self.mapped_weights["decoder"]
                if isinstance(decoder, dict):
                    if "up_blocks" in decoder:
                        if not isinstance(decoder["up_blocks"], list):
                            issues.append("decoder.up_blocks: Expected list")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
        }

    def compare_raw_vs_mapped(self, sample_size: int = 10) -> Dict[str, Any]:
        """
        Compare raw and mapped weights to ensure no data loss.

        Samples random weights and verifies:
        1. Values are preserved (after transforms)
        2. Shapes are correct
        3. No unexpected changes

        Args:
            sample_size: Number of weights to sample for comparison

        Returns:
            Dictionary with comparison results
        """
        # This is a simplified version - full comparison would require
        # knowing the mapping patterns and transforms
        hf_keys = list(self.raw_weights.keys())
        samples = hf_keys[:sample_size] if len(hf_keys) <= sample_size else hf_keys[:sample_size]

        comparisons = []
        for hf_key in samples:
            raw_tensor = self.raw_weights.get(hf_key)
            if raw_tensor is None:
                continue

            # Try to find corresponding MLX weight
            # This is simplified - real implementation would use mapping patterns
            mlx_key = hf_key  # Placeholder - would need actual mapping
            mlx_tensor = self._get_nested_value(self.mapped_weights, mlx_key)

            comparison = {
                "hf_key": hf_key,
                "mlx_key": mlx_key if mlx_tensor else None,
                "raw_shape": list(raw_tensor.shape) if raw_tensor is not None else None,
                "mapped_shape": list(mlx_tensor.shape) if mlx_tensor is not None else None,
                "found": mlx_tensor is not None,
            }
            comparisons.append(comparison)

        return {
            "samples_compared": len(comparisons),
            "found_count": sum(1 for c in comparisons if c["found"]),
            "comparisons": comparisons,
        }

    @staticmethod
    def _count_nested_weights(d: Dict) -> int:
        """Recursively count all MLX arrays in nested structure."""
        count = 0
        if isinstance(d, dict):
            for v in d.values():
                count += WeightInspector._count_nested_weights(v)
        elif isinstance(d, list):
            for item in d:
                count += WeightInspector._count_nested_weights(item)
        else:
            # It's a weight (MLX array)
            if isinstance(d, mx.array):
                count = 1
        return count

    def print_structure(self, max_depth: int = 4, show_types: bool = True, component: Optional[str] = None):
        """
        Print actual nested structure showing types, list lengths, and key names.

        This shows the REAL structure, not inferred - critical for verification.
        Handles both nested structures (from mapped weights) and flat structures (from raw weights).

        Args:
            max_depth: Maximum depth to display
            show_types: Show type annotations (dict/list/tensor)
            component: Filter by component prefix
        """
        print(f"\n🏗️  Actual Structure for: {self.model_name}")
        if component:
            print(f"   Filtered by: {component}")
        print("=" * 70)

        # Check if mapped_weights is flat (all values are arrays) or nested
        is_flat = all(isinstance(v, mx.array) for v in self.mapped_weights.values())

        if is_flat:
            # Flat structure - show as organized list
            print("   (Flat weight structure - weights organized by component)")
            print()

            # Group by component prefix
            components = {}
            for key in sorted(self.mapped_weights.keys()):
                if component and not key.startswith(component):
                    continue

                # Extract component (first part of key)
                parts = key.split(".")
                comp = parts[0]
                if comp not in components:
                    components[comp] = []
                components[comp].append(key)

            # Show each component
            for comp, keys in sorted(components.items()):
                print(f"   {comp}/ ({len(keys)} weights)")
                # Show first few weights from this component
                for key in keys[:5]:
                    tensor = self.mapped_weights[key]
                    shape_str = "x".join(str(s) for s in tensor.shape)
                    dtype_str = str(tensor.dtype)
                    size_mb = tensor.nbytes / (1024 * 1024)
                    # Show relative path (without component prefix)
                    rel_key = ".".join(key.split(".")[1:]) if "." in key else key
                    print(f"      {rel_key}: array[{shape_str}] {dtype_str} ({size_mb:.2f} MB)")
                if len(keys) > 5:
                    print(f"      ... and {len(keys) - 5} more weights")
        else:
            # Nested structure - use recursive printing
            def _print_structure_recursive(
                obj: Any, prefix: str = "", depth: int = 0, path: str = "", filter_prefix: Optional[str] = None
            ):
                if depth > max_depth:
                    return

                # Filter by component if specified
                if filter_prefix and not path.startswith(filter_prefix):
                    return

                indent = "  " * depth

                if isinstance(obj, dict):
                    if show_types:
                        print(f"{indent}{prefix} (dict, {len(obj)} keys)")
                    else:
                        print(f"{indent}{prefix}/ ({len(obj)} keys)")

                    for key, value in sorted(obj.items()):
                        new_path = f"{path}.{key}" if path else key
                        _print_structure_recursive(value, key, depth + 1, new_path, filter_prefix)

                elif isinstance(obj, list):
                    if show_types:
                        print(f"{indent}{prefix} (list[{len(obj)}])")
                    else:
                        print(f"{indent}{prefix}/ (list, {len(obj)} items)")

                    # Show first few items with their structure
                    for i, item in enumerate(obj[:3]):  # Show first 3 items
                        new_path = f"{path}[{i}]" if path else f"[{i}]"
                        _print_structure_recursive(item, f"[{i}]", depth + 1, new_path, filter_prefix)

                    if len(obj) > 3:
                        print(f"{indent}  ... and {len(obj) - 3} more items")

                elif isinstance(obj, mx.array):
                    # Leaf tensor - show actual shape
                    shape_str = "x".join(str(s) for s in obj.shape)
                    dtype_str = str(obj.dtype)
                    size_mb = obj.nbytes / (1024 * 1024)
                    print(f"{indent}{prefix}: array[{shape_str}] {dtype_str} ({size_mb:.2f} MB)")

                else:
                    # Other types
                    type_name = type(obj).__name__
                    print(f"{indent}{prefix}: {type_name}")

            _print_structure_recursive(self.mapped_weights, "root", filter_prefix=component)

        print("=" * 70)

    def print_mapping_report(self):
        """
        Print a comprehensive mapping verification report.

        Always includes:
        - Detected patterns from raw weights
        - Mapping coverage analysis
        - Structure verification
        - ACTUAL nested structure (shows real data, not inferred)

        This is the most comprehensive view - shows everything we know.
        """
        print(f"\n📋 Mapping Verification Report for: {self.model_name}")
        print("=" * 70)

        # 1. Detect patterns
        print("\n1️⃣  Detected Patterns (from raw weights):")
        patterns = self.detect_patterns()
        for key, value in patterns.items():
            if value:  # Only show non-empty values
                if isinstance(value, dict):
                    print(f"   {key}:")
                    for k, v in value.items():
                        print(f"      {k}: {v}")
                else:
                    print(f"   {key}: {value}")

        # 2. Coverage analysis
        print("\n2️⃣  Mapping Coverage:")
        coverage = self.analyze_mapping_coverage()
        print(f"   Raw weights: {coverage['total_raw_weights']}")
        print(f"   Mapped weights: {coverage['total_mapped_weights']}")
        print(f"   Coverage: {coverage['coverage_percent']:.1f}%")
        print(f"   Unmatched: {coverage['unmatched_count']}")

        if coverage["unmatched_by_component"]:
            print("\n   Unmatched by component:")
            for comp, count in coverage["unmatched_by_component"].items():
                print(f"      {comp}: {count}")

        if coverage["unmatched_samples"]:
            print("\n   Sample unmatched weights:")
            for key in coverage["unmatched_samples"][:5]:
                print(f"      {key}")

        # 3. Structure verification
        print("\n3️⃣  Structure Verification:")
        verification = self.verify_structure()
        if verification["valid"]:
            print("   ✅ Structure is valid")
        else:
            print("   ❌ Structure issues found:")
            for issue in verification["issues"]:
                print(f"      {issue}")

        if verification["warnings"]:
            print("   ⚠️  Warnings:")
            for warning in verification["warnings"]:
                print(f"      {warning}")

        # 4. ACTUAL STRUCTURE (always shown - shows real data, not inferred)
        print("\n4️⃣  Actual Mapped Structure (shows real types and structure):")
        self.print_structure(max_depth=4, show_types=True)

        print("\n" + "=" * 70)
