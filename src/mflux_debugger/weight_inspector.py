"""
Weight Inspector - Inspect and visualize model weights.

Provides tools to inspect raw HuggingFace weights and mapped MLX weights,
with support for searching, filtering, and displaying tensor statistics.
"""

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
