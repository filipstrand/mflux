"""
Tensor Debug - Convenient API for saving/loading tensors between PyTorch and MLX implementations.

This module provides utilities for saving tensors from PyTorch and loading them in MLX,
enabling easy comparison and debugging of model implementations.

KEY FEATURE: debug_save() and debug_load() work from ANYWHERE!
- Works from mflux repository code
- Works from external repositories (diffusers, transformers, etc.)
- Works from scripts in any location
- Automatically handles path resolution - no configuration needed!

Typical workflows:

1. Simple case (single tensor):
   PyTorch: debug_save(tensor, "tensor_name", exit_after=True)
   MLX:     tensor = debug_load("tensor_name")

2. Loop case (multiple iterations):
   PyTorch: debug_save(hidden_states, f"hidden_states_{block}_{timestep}")
   MLX:     hidden_states = debug_load(f"hidden_states_{block}_{timestep}")

The debug functions are designed to be called inline with minimal code changes.
You don't need to worry about paths - just import and use!
"""

import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    import mlx.core as mx
except ImportError:
    mx = None  # MLX not available (PyTorch-only environment)

from mflux_debugger.image_tensor_paths import (
    get_tensors_archive_dir,
    get_tensors_latest_dir,
)

logger = logging.getLogger(__name__)


def _get_debug_tensors_dir() -> Path:
    """
    Get the debug tensors directory (latest/ subdirectory).

    This function is called dynamically to ensure path resolution works correctly
    regardless of where the code is imported from or when it's called.
    """
    return get_tensors_latest_dir()


def _get_debug_tensors_archive_dir() -> Path:
    """
    Get the debug tensors archive directory.

    This function is called dynamically to ensure path resolution works correctly.
    """
    return get_tensors_archive_dir()


# Note: All code should use _get_debug_tensors_dir() and _get_debug_tensors_archive_dir()
# instead of these constants to ensure dynamic path resolution

# Size limits (in GB)
SIZE_WARNING_GB = 1.0  # Warn when directory exceeds this size
SIZE_LIMIT_GB = 10.0  # Block saves when directory exceeds this size


def _get_directory_size_gb() -> float:
    """Get the total size of the debug tensors directory in GB."""
    total_bytes = sum(f.stat().st_size for f in _get_debug_tensors_dir().glob("*.npy"))
    return total_bytes / (1024**3)


def _check_directory_size(operation: str = "save") -> None:
    """
    Check if the debug tensors directory is getting too large.

    Args:
        operation: Either "save" (for debug_save) or "load" (for debug_load/debugger start)

    Raises:
        RuntimeError: If directory size exceeds SIZE_LIMIT_GB
    """
    size_gb = _get_directory_size_gb()

    if size_gb >= SIZE_LIMIT_GB:
        tensors_dir = _get_debug_tensors_dir()
        error_msg = (
            f"🚫 ERROR: Debug tensors directory is {size_gb:.2f} GB "
            f"(limit: {SIZE_LIMIT_GB} GB)\n"
            f"   Directory: {tensors_dir}\n"
            f"   Please clean up old tensors using:\n"
            f"     from mflux_debugger.tensor_debug import debug_clear\n"
            f"     debug_clear()  # Clear all tensors\n"
            f"   Or manually delete files in: {tensors_dir}"
        )
        logger.error(error_msg)
        print(error_msg, flush=True)
        raise RuntimeError(f"Debug tensors directory exceeds {SIZE_LIMIT_GB} GB limit")

    elif size_gb >= SIZE_WARNING_GB:
        tensors_dir = _get_debug_tensors_dir()
        warning_msg = (
            f"⚠️  WARNING: Debug tensors directory is {size_gb:.2f} GB "
            f"(warning threshold: {SIZE_WARNING_GB} GB)\n"
            f"   Directory: {tensors_dir}\n"
            f"   Consider cleaning up old tensors to free disk space.\n"
            f"   Use: debug_clear() to remove all tensors"
        )
        logger.warning(warning_msg)
        print(warning_msg, flush=True)


def debug_full_cleanup() -> None:
    """
    Archive the entire mflux_debugger folder to start fresh.

    This function archives all debug logs, checkpoints, tensors, and images
    from the mflux_debugger/ directory to mflux_debugger_deleted/ with a timestamp.
    Use this at the beginning of scripts to ensure a clean debugging session
    while preserving previous debug data.

    Examples:
        ```python
        from mflux_debugger.tensor_debug import debug_full_cleanup

        # At the beginning of your script
        debug_full_cleanup()

        # Then proceed with your debugging code
        ```
    """
    # Get the mflux_debugger directory path (top-level, not in src)
    src_dir = Path(__file__).parent  # src/mflux_debugger
    repo_root = src_dir.parent.parent  # /Users/filipstrand/Desktop/mflux
    debugger_dir = repo_root / "mflux_debugger"

    if not debugger_dir.exists():
        logger.debug(f"mflux_debugger/ directory doesn't exist at {debugger_dir} - nothing to clean")
        return

    # Count files before archiving for logging
    files = list(debugger_dir.rglob("*"))
    file_count = sum(1 for f in files if f.is_file())
    total_bytes = sum(f.stat().st_size for f in files if f.is_file())
    size_mb = total_bytes / (1024 * 1024)

    # Create archive directory at the same level as mflux_debugger
    archive_base_dir = repo_root / "mflux_debugger_deleted"
    archive_base_dir.mkdir(exist_ok=True)

    # Create timestamped archive directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = archive_base_dir / f"mflux_debugger_{timestamp}"

    # Move the entire mflux_debugger directory to the archive
    # We need to move the contents, not the directory itself (to preserve the mflux_debugger name)
    archive_dir.mkdir(parents=True, exist_ok=True)

    # Move all contents to the archive
    for item in debugger_dir.iterdir():
        dest = archive_dir / item.name
        if item.is_file():
            shutil.move(str(item), str(dest))
        elif item.is_dir():
            shutil.move(str(item), str(dest))

    logger.info(f"📦 Archived mflux_debugger/ directory: {file_count} files, {size_mb:.2f} MB → {archive_dir.name}")
    print(f"📦 Archived mflux_debugger/ directory: {file_count} files, {size_mb:.2f} MB", flush=True)
    print(f"   Archive: {archive_dir}", flush=True)


def debug_save(tensor: Any, name: str, exit_after: bool = False, skip: bool = False) -> None:
    """
    Save a tensor for later loading with debug_load().

    This function works from ANYWHERE - you don't need to worry about paths or locations.
    It automatically finds the correct mflux_debugger directory regardless of where it's called from.

    Key Features:
    - Works from mflux repository code
    - Works from external repositories (like diffusers, transformers, etc.)
    - Works from scripts in any location
    - Automatically handles path resolution - you just call it!

    The tensor is saved to mflux_debugger/tensors/latest/{name}.npy and can be loaded
    later using debug_load(name) from anywhere.

    Args:
        tensor: Tensor to save (PyTorch tensor, MLX array, or numpy array)
        name: Unique name for the saved tensor (alphanumeric + underscores recommended)
              Can use f-strings for loop indices: f"hidden_states_{block}_{timestep}"
        exit_after: If True, exit the program after saving (useful to capture early tensors)
        skip: If True, skip saving entirely (useful for conditional saving)

    Examples:
        ```python
        # Import once at the top of your file
        from mflux_debugger.tensor_debug import debug_save

        # Then use it ANYWHERE - no path manipulation needed!
        class MyModel:
            def forward(self, x):
                # Simple case - save one tensor
                latents = self.prepare_latents(...)
                debug_save(latents, "initial_latents")

                # Conditional saving
                debug_save(latents, "debug_latents", skip=not DEBUG_MODE)

                # Loop case - save tensor at each iteration
                for block in range(num_blocks):
                    for timestep in range(num_timesteps):
                        hidden_states = self.blocks[block](hidden_states, t=timestep)
                        debug_save(hidden_states, f"hidden_states_{block}_{timestep}")
        ```

    Note:
        - The tensor is automatically converted to a format compatible with both PyTorch and MLX
        - For complex tensors, saves both real and imaginary parts in a structured array
        - All tensors are saved to the same location (mflux_debugger/tensors/latest/)
        - Multiple saves with the same name in one session will overwrite
        - For loops: use f-strings with indices to create unique names per iteration
        - The function handles all path resolution automatically - you don't need to configure anything!
    """
    # Get caller information for logging
    import inspect

    frame = inspect.currentframe().f_back
    caller_file = frame.f_code.co_filename
    caller_line = frame.f_lineno
    caller_location = f"{caller_file}:{caller_line}"

    # Skip saving if requested
    if skip:
        logger.debug(f"⏭️  [tensor_debug] Skipped saving '{name}' (skip=True) at {caller_location}")
        return

    try:
        # Check directory size before saving
        _check_directory_size(operation="save")

        # Handle MLX arrays
        if mx is not None and isinstance(tensor, mx.array):
            # Convert MLX array to numpy, then to PyTorch tensor for consistent handling
            tensor_np = np.array(tensor)
            tensor_cpu = torch.from_numpy(tensor_np) if torch is not None else tensor_np
        # Handle PyTorch tensors
        elif torch is not None and isinstance(tensor, torch.Tensor):
            # Detach and move to CPU
            tensor_cpu = tensor.detach().cpu()
        # Handle numpy arrays
        elif isinstance(tensor, np.ndarray):
            tensor_cpu = torch.from_numpy(tensor) if torch is not None else tensor
        else:
            raise TypeError(f"Expected PyTorch tensor, MLX array, or numpy array, got {type(tensor).__name__}")

        # Check if tensor is complex (PyTorch tensors only)
        if torch is not None and isinstance(tensor_cpu, torch.Tensor):
            is_complex = tensor_cpu.is_complex()
        else:
            # For numpy arrays, check dtype
            is_complex = np.iscomplexobj(tensor_cpu)

        if is_complex:
            # For complex tensors, save as structured array with 'real' and 'imag' fields
            # This preserves both components for proper RoPE embedding reconstruction
            if torch is not None and isinstance(tensor_cpu, torch.Tensor):
                real_part = tensor_cpu.real.float().numpy()
                imag_part = tensor_cpu.imag.float().numpy()
            else:
                # Numpy array
                real_part = np.real(tensor_cpu).astype(np.float32)
                imag_part = np.imag(tensor_cpu).astype(np.float32)

            # Create structured array with named fields
            tensor_np = np.empty(real_part.shape, dtype=[("real", "f4"), ("imag", "f4")])
            tensor_np["real"] = real_part
            tensor_np["imag"] = imag_part

            info_str = "complex (real+imag)"
        else:
            # For real tensors, preserve dtype (int32/int64 for indices, float32 for values)
            # Only convert to float if it's already a floating-point type
            if torch is not None and isinstance(tensor_cpu, torch.Tensor):
                original_dtype = tensor_cpu.dtype
                if original_dtype in (
                    torch.int32,
                    torch.int64,
                    torch.long,
                    torch.int16,
                    torch.int8,
                    torch.uint8,
                ):
                    # Preserve integral types (important for input_ids, attention_mask, etc.)
                    tensor_np = tensor_cpu.numpy()
                else:
                    # Convert floating-point types to float32
                    tensor_np = tensor_cpu.float().numpy()
            else:
                # Numpy array - preserve dtype for integers, convert floats to float32
                if np.issubdtype(tensor_cpu.dtype, np.integer):
                    tensor_np = tensor_cpu
                else:
                    tensor_np = tensor_cpu.astype(np.float32)
            info_str = "real"

        # Ensure directory exists (always create if needed - transparent to user)
        # This is necessary because the directory might be deleted by debug_full_cleanup(),
        # so we need to recreate it dynamically
        tensors_dir = _get_debug_tensors_dir()
        tensors_dir.mkdir(parents=True, exist_ok=True)

        # Save tensor (no versioning - sessions are isolated via archiving)
        save_path = tensors_dir / f"{name}.npy"

        # Save tensor
        np.save(save_path, tensor_np)

        # Get tensor info
        shape = tensor_np.shape
        dtype = tensor_np.dtype
        size_mb = tensor_np.nbytes / (1024 * 1024)

        logger.info(f"✅ Tensor saved: '{name}' → {save_path.name}")
        logger.info(f"   Shape: {shape}, dtype: {dtype}, type: {info_str}, size: {size_mb:.2f} MB")
        logger.info(f"   Called from: {caller_location}")
        print(f"✅ [tensor_debug] Saved '{name}': shape={shape}, type={info_str}, size={size_mb:.2f}MB", flush=True)
        print(f"   File: {save_path}", flush=True)
        print(f"   Called from: {caller_location}", flush=True)

        # Exit if requested (useful for capturing early tensors)
        if exit_after:
            print("🛑 [tensor_debug] Exiting after save as requested (exit_after=True)", flush=True)
            sys.exit(0)

    except Exception as e:
        logger.error(f"❌ Failed to save tensor '{name}': {e}")
        raise RuntimeError(f"debug_save failed for '{name}': {e}") from e


def debug_load(
    name: str,
    exact_version: bool = False,
    to: str = "mlx",
    device: Optional[Any] = None,
    dtype: Optional[Any] = None,
) -> Any:
    """
    Load a tensor saved with debug_save().

    This function works from ANYWHERE - you don't need to worry about paths or locations.
    It automatically finds the correct mflux_debugger directory and loads the tensor
    that was saved with debug_save(name).

    Key Features:
    - Works from mflux repository code
    - Works from external repositories (like diffusers, transformers, etc.)
    - Works from scripts in any location
    - Automatically handles path resolution - you just call it!

    Args:
        name: Name of the saved tensor (without .npy extension)
              Must match the name used in debug_save(name)
              Can use f-strings for loop indices: f"hidden_states_{block}_{timestep}"
        exact_version: If True, load the exact name without auto-versioning (default: False)
        to: Target framework format - "mlx" (default) or "pytorch"
        device: Optional device for PyTorch tensors (e.g., "cuda", "cpu", torch.device("cuda:0"))
                Ignored for MLX (MLX handles device automatically)
        dtype: Optional dtype for PyTorch tensors (e.g., torch.bfloat16, torch.float32)
               For MLX, can be a string like "float32" or "bfloat16"

    Returns:
        MLX array (if to="mlx", default)
        PyTorch tensor (if to="pytorch")
        For complex tensors, returns a tuple (real, imag) in the requested format

    Raises:
        FileNotFoundError: If the tensor file is not found
        ValueError: If to parameter is not "mlx" or "pytorch"

    Examples:
        ```python
        # Import once at the top of your file
        from mflux_debugger.tensor_debug import debug_load

        # Then use it ANYWHERE - no path manipulation needed!
        class MyModel:
            def forward(self, x):
                # Simple case - load one tensor as MLX (default)
                latents = debug_load("initial_latents")
                # Or explicitly:
                latents = debug_load("initial_latents", to="mlx")

                # Load as PyTorch tensor with device and dtype
                latents_pt = debug_load("initial_latents", to="pytorch", device="cuda", dtype=torch.bfloat16)

                # Load with device from context
                latents = debug_load("initial_latents", to="pytorch", device=self.device, dtype=torch.bfloat16)

                # Loop case - load tensor at each iteration
                for block in range(num_blocks):
                    for timestep in range(num_timesteps):
                        hidden_states = debug_load(f"hidden_states_{block}_{timestep}")

                # Complex tensor case (e.g., RoPE embeddings)
                rope_real, rope_imag = debug_load("rope_embeddings")  # Returns (real, imag) tuple
        ```

    Note:
        - Automatically converts to the requested format (MLX array or PyTorch tensor)
        - For complex tensors (saved from PyTorch complex tensors), returns tuple (real, imag)
        - If device/dtype are provided, applies them after conversion
        - Raises FileNotFoundError if tensor not found (fail-fast behavior)
        - For loops: use f-strings with indices matching those used in debug_save()
        - The function handles all path resolution automatically - you don't need to configure anything!
    """
    # Get caller information for logging
    import inspect

    frame = inspect.currentframe().f_back
    caller_file = frame.f_code.co_filename
    caller_line = frame.f_lineno
    caller_location = f"{caller_file}:{caller_line}"

    # Validate 'to' parameter
    if to not in ("mlx", "pytorch"):
        raise ValueError(f"Invalid 'to' parameter: {to}. Must be 'mlx' or 'pytorch'")

    # Check required frameworks
    if to == "mlx" and mx is None:
        raise ImportError("MLX not available - debug_load requires MLX to be installed when to='mlx'")
    if to == "pytorch" and torch is None:
        raise ImportError("PyTorch not available - debug_load requires PyTorch to be installed when to='pytorch'")

    try:
        # Check directory size (informational only for load)
        _check_directory_size(operation="load")

        # Load tensor file (no versioning - sessions are isolated)
        tensors_dir = _get_debug_tensors_dir()
        load_path = tensors_dir / f"{name}.npy"

        # Check if file exists; if missing, return None so callers can fall back
        if not load_path.exists():
            available_files = sorted([f.stem for f in tensors_dir.glob("*.npy")])
            warn_msg = f"⚠️ Tensor '{name}' not found in {tensors_dir}"
            if available_files:
                warn_msg += f"\n   Available tensors: {', '.join(available_files[:10])}"
                if len(available_files) > 10:
                    warn_msg += f" ... and {len(available_files) - 10} more"
            logger.warning(warn_msg)
            print(warn_msg, flush=True)
            return None

        # Load NumPy array
        tensor_np = np.load(load_path)

        # Check if this is a complex tensor (structured array with 'real' and 'imag' fields)
        is_complex = (
            tensor_np.dtype.names is not None and "real" in tensor_np.dtype.names and "imag" in tensor_np.dtype.names
        )

        if is_complex:
            # Extract real and imaginary parts
            real_np = tensor_np["real"]
            imag_np = tensor_np["imag"]

            # Convert to requested format
            if to == "mlx":
                real_tensor = mx.array(real_np)
                imag_tensor = mx.array(imag_np)
                # Apply dtype if specified (MLX)
                if dtype is not None:
                    if isinstance(dtype, str):
                        dtype_map = {
                            "float32": mx.float32,
                            "float16": mx.float16,
                            "bfloat16": mx.bfloat16,
                        }
                        if dtype in dtype_map:
                            real_tensor = mx.astype(real_tensor, dtype_map[dtype])
                            imag_tensor = mx.astype(imag_tensor, dtype_map[dtype])
                        else:
                            logger.warning(f"Unknown MLX dtype string '{dtype}', ignoring")
                    else:
                        real_tensor = mx.astype(real_tensor, dtype)
                        imag_tensor = mx.astype(imag_tensor, dtype)
            else:  # to == "pytorch"
                real_tensor = torch.from_numpy(real_np).float()
                imag_tensor = torch.from_numpy(imag_np).float()

                # Apply device and dtype if specified (PyTorch)
                if device is not None or dtype is not None:
                    if device is not None and dtype is not None:
                        real_tensor = real_tensor.to(device=device, dtype=dtype)
                        imag_tensor = imag_tensor.to(device=device, dtype=dtype)
                    elif device is not None:
                        real_tensor = real_tensor.to(device=device)
                        imag_tensor = imag_tensor.to(device=device)
                    elif dtype is not None:
                        real_tensor = real_tensor.to(dtype=dtype)
                        imag_tensor = imag_tensor.to(dtype=dtype)

            # Get tensor info
            shape = real_tensor.shape
            result_dtype = real_tensor.dtype if hasattr(real_tensor, "dtype") else type(real_tensor).__name__
            size_mb = (tensor_np.nbytes) / (1024 * 1024)

            device_info = f", device={device}" if device is not None else ""
            dtype_info = f", dtype={dtype}" if dtype is not None else ""
            logger.info(f"✅ Tensor loaded: '{name}' ← {load_path.name} (to={to}{device_info}{dtype_info})")
            logger.info(f"   Shape: {shape}, dtype={result_dtype} (complex: real+imag), size: {size_mb:.2f} MB")
            logger.info(f"   Called from: {caller_location}")
            print(
                f"✅ [tensor_debug] Loaded '{name}': shape={shape}, type=complex (real+imag), size={size_mb:.2f}MB, to={to}{device_info}{dtype_info}",
                flush=True,
            )
            print(f"   File: {load_path}", flush=True)
            print(f"   Called from: {caller_location}", flush=True)

            # Return as tuple (real, imag) for complex tensors
            return (real_tensor, imag_tensor)
        else:
            # For real tensors, handle dtype conversion
            # If saved as float32 but contains integer values (common for input_ids, attention_mask),
            # detect and convert appropriately
            if tensor_np.dtype == np.float32:
                # Check if all values are integers (within float32 precision)
                if np.allclose(tensor_np, np.round(tensor_np)):
                    # Convert to int32 for MLX, int64 for PyTorch
                    if to == "mlx":
                        tensor_np = tensor_np.astype(np.int32)
                    else:  # to == "pytorch"
                        tensor_np = tensor_np.astype(np.int64)

            # Convert to requested format
            if to == "mlx":
                tensor_result = mx.array(tensor_np)
                # Apply dtype if specified (MLX)
                if dtype is not None:
                    if isinstance(dtype, str):
                        # Convert string dtype to MLX dtype
                        dtype_map = {
                            "float32": mx.float32,
                            "float16": mx.float16,
                            "bfloat16": mx.bfloat16,
                            "int32": mx.int32,
                            "int64": mx.int64,
                        }
                        if dtype in dtype_map:
                            tensor_result = mx.astype(tensor_result, dtype_map[dtype])
                        else:
                            logger.warning(f"Unknown MLX dtype string '{dtype}', ignoring")
                    else:
                        # Assume it's already an MLX dtype
                        tensor_result = mx.astype(tensor_result, dtype)
            else:  # to == "pytorch"
                # Convert numpy to PyTorch tensor
                if np.issubdtype(tensor_np.dtype, np.integer):
                    tensor_result = torch.from_numpy(tensor_np).long()
                else:
                    tensor_result = torch.from_numpy(tensor_np).float()

                # Apply device and dtype if specified (PyTorch)
                if device is not None or dtype is not None:
                    if device is not None and dtype is not None:
                        tensor_result = tensor_result.to(device=device, dtype=dtype)
                    elif device is not None:
                        tensor_result = tensor_result.to(device=device)
                    elif dtype is not None:
                        tensor_result = tensor_result.to(dtype=dtype)

        # Get tensor info
        shape = tensor_result.shape
        result_dtype = tensor_result.dtype if hasattr(tensor_result, "dtype") else type(tensor_result).__name__
        size_mb = tensor_np.nbytes / (1024 * 1024)

        device_info = f", device={device}" if device is not None else ""
        dtype_info = f", dtype={dtype}" if dtype is not None else ""
        logger.info(f"✅ Tensor loaded: '{name}' ← {load_path.name} (to={to}{device_info}{dtype_info})")
        logger.info(f"   Shape: {shape}, dtype: {result_dtype}, size: {size_mb:.2f} MB")
        logger.info(f"   Called from: {caller_location}")
        print(
            f"✅ [tensor_debug] Loaded '{name}': shape={shape}, size={size_mb:.2f}MB, to={to}{device_info}{dtype_info}",
            flush=True,
        )
        print(f"   File: {load_path}", flush=True)
        print(f"   Called from: {caller_location}", flush=True)

        return tensor_result
    except Exception as e:
        logger.error(f"❌ Failed to load tensor '{name}': {e}")
        raise RuntimeError(f"debug_load failed for '{name}': {e}") from e


def debug_list(pattern: Optional[str] = None) -> list[str]:
    """
    List all available debug tensors, optionally filtered by pattern.

    Args:
        pattern: Optional glob pattern to filter tensor names (e.g., "hidden_states_*")

    Returns:
        List of tensor names (without .npy extension)

    Examples:
        ```python
        # At the top of your file (or in a Python REPL):
        from mflux_debugger.tensor_debug import debug_list

        # List all tensors
        all_tensors = debug_list()

        # List tensors for a specific pattern
        block_0_tensors = debug_list("hidden_states_0_*")
        timestep_5_tensors = debug_list("*_5")
        ```
    """
    search_pattern = f"{pattern}.npy" if pattern else "*.npy"
    return [file.stem for file in sorted(_get_debug_tensors_dir().glob(search_pattern))]


def debug_info(name: str) -> dict[str, Any]:
    """
    Get information about a debug tensor without loading it.

    Args:
        name: Name of the saved tensor

    Returns:
        Dictionary with shape, dtype, size, and file path

    Example:
        ```python
        # At the top of your file (or in a Python REPL):
        from mflux_debugger.tensor_debug import debug_info

        # Use it:
        info = debug_info("initial_latents")
        print(f"Shape: {info['shape']}, Size: {info['size_mb']:.2f} MB")
        ```
    """
    # Get tensor path
    tensors_dir = _get_debug_tensors_dir()
    tensor_path = tensors_dir / f"{name}.npy"

    if not tensor_path.exists():
        raise FileNotFoundError(f"Tensor '{name}' not found in {tensors_dir}")

    # Load just the header to get metadata (efficient)
    try:
        with open(tensor_path, "rb") as f:
            np.lib.format.read_magic(f)  # Read magic bytes but don't need version
            # Try new API first, fall back to loading full array
            try:
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(f)
            except (AttributeError, ValueError):
                # Fall back to loading the array to get metadata
                arr = np.load(tensor_path)
                shape = arr.shape
                dtype = arr.dtype
    except (OSError, ValueError):
        # Ultimate fallback - just load it
        arr = np.load(tensor_path)
        shape = arr.shape
        dtype = arr.dtype

    size_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    size_mb = size_bytes / (1024 * 1024)

    return {
        "name": name,
        "shape": shape,
        "dtype": str(dtype),
        "size_mb": size_mb,
        "file": str(tensor_path),
    }


def debug_clear(name: Optional[str] = None, confirm: bool = True) -> int:
    """
    Clear debug tensors from disk.

    Args:
        name: Specific tensor name to clear (None = clear all)
        confirm: If True, require confirmation before deleting (default: True)

    Returns:
        Number of files deleted

    Example:
        ```python
        # At the top of your file (or in a Python REPL):
        from mflux_debugger.tensor_debug import debug_clear

        # Clear specific tensor
        debug_clear("initial_latents", confirm=False)

        # Clear all tensors (with confirmation)
        debug_clear()
        ```
    """
    # Find files to delete
    tensors_dir = _get_debug_tensors_dir()
    if name:
        # Clear specific tensor and its versions
        files_to_delete = list(tensors_dir.glob(f"{name}*.npy"))
    else:
        # Clear all tensors
        files_to_delete = list(tensors_dir.glob("*.npy"))

    if not files_to_delete:
        logger.info("No tensors found to clear")
        return 0

    # Show what will be deleted
    total_size = sum(f.stat().st_size for f in files_to_delete)
    size_mb = total_size / (1024 * 1024)

    print(f"🗑️  Found {len(files_to_delete)} file(s) to delete ({size_mb:.2f} MB)")
    for f in files_to_delete:
        print(f"   - {f.name}")

    # Confirm if requested
    if confirm:
        response = input("\n⚠️  Delete these files? [y/N]: ").strip().lower()
        if response != "y":
            print("❌ Cancelled")
            return 0

    # Delete files
    deleted = 0
    for f in files_to_delete:
        try:
            f.unlink()
            deleted += 1
        except Exception as e:  # noqa: BLE001, PERF203
            logger.warning(f"Failed to delete {f.name}: {e}")

    logger.info(f"✅ Deleted {deleted} file(s)")
    print(f"✅ Deleted {deleted} file(s)")
    return deleted


def archive_tensors(script_name: Optional[str] = None) -> int:
    """
    Archive existing debug tensors to a timestamped directory.

    Called automatically when starting a new debug session to preserve tensors
    from previous runs while clearing the active tensors directory.

    Args:
        script_name: Optional script name to include in archive directory name

    Returns:
        Number of files archived

    Example:
        ```python
        from mflux_debugger.tensor_debug import archive_tensors

        # Manual archiving (usually automatic)
        archived_count = archive_tensors("my_script")
        ```
    """
    # Find all tensor files
    tensor_files = list(_get_debug_tensors_dir().glob("*.npy"))

    if not tensor_files:
        logger.debug("No tensors to archive")
        return 0

    # Create timestamped archive directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_base = _get_debug_tensors_archive_dir()
    if script_name:
        archive_dir = archive_base / f"{script_name}_{timestamp}"
    else:
        archive_dir = archive_base / f"tensors_{timestamp}"

    archive_dir.mkdir(parents=True, exist_ok=True)

    # Move files to archive
    archived = 0
    total_size = 0
    for tensor_file in tensor_files:
        try:
            total_size += tensor_file.stat().st_size
            dest = archive_dir / tensor_file.name
            shutil.move(str(tensor_file), str(dest))
            archived += 1
        except Exception as e:  # noqa: BLE001, PERF203
            logger.warning(f"Failed to archive {tensor_file.name}: {e}")

    if archived > 0:
        size_mb = total_size / (1024 * 1024)
        # Get relative path from debugger root
        from mflux_debugger.log_paths import get_debugger_root

        debugger_root = get_debugger_root()
        archive_rel = archive_dir.relative_to(debugger_root.parent)
        logger.info(f"📦 Archived {archived} tensor(s) ({size_mb:.2f} MB) to {archive_rel}")
        print(f"📦 Archived {archived} tensor(s) ({size_mb:.2f} MB)")
        print(f"   Archive: {archive_rel}")

    return archived


# Convenience function for debugging - print all available tensors
def debug_show_all() -> None:
    """Print debug information about all saved debug tensors."""
    tensors = debug_list()
    size_gb = _get_directory_size_gb()

    tensors_dir = _get_debug_tensors_dir()
    if not tensors:
        print("📋 No debug tensors found")
        print(f"   Storage: {tensors_dir}")
        return

    print(f"📋 Debug tensors ({len(tensors)}):")
    print(f"   Storage: {tensors_dir}")
    print(f"   Total size: {size_gb:.3f} GB")

    # Show warning if size is concerning
    if size_gb >= SIZE_WARNING_GB:
        print(f"   ⚠️  Warning: Directory size exceeds {SIZE_WARNING_GB} GB threshold")
    if size_gb >= SIZE_LIMIT_GB:
        print(f"   🚫 ERROR: Directory size exceeds {SIZE_LIMIT_GB} GB limit!")
    print()

    for name in tensors:
        try:
            info = debug_info(name)
            print(f"   • {name}")
            print(f"     Shape: {info['shape']}, dtype: {info['dtype']}")
            print(f"     Size: {info['size_mb']:.2f} MB")
        except Exception as e:  # noqa: BLE001, PERF203
            print(f"   • {name} (error: {e})")


def check_debug_directory_on_startup() -> None:
    """
    Check debug tensors directory size at debugger startup.
    Called automatically by debugging tools.

    Raises:
        RuntimeError: If directory size exceeds SIZE_LIMIT_GB
    """
    size_gb = _get_directory_size_gb()

    if size_gb >= SIZE_LIMIT_GB:
        tensors_dir = _get_debug_tensors_dir()
        error_msg = (
            f"\n{'=' * 70}\n"
            f"🚫 DEBUGGER STARTUP BLOCKED\n"
            f"{'=' * 70}\n"
            f"Debug tensors directory is {size_gb:.2f} GB (limit: {SIZE_LIMIT_GB} GB)\n"
            f"Directory: {tensors_dir}\n\n"
            f"Please clean up before starting the debugger:\n"
            f"  1. From Python:\n"
            f"     >>> from mflux_debugger.tensor_debug import debug_clear\n"
            f"     >>> debug_clear()\n\n"
            f"  2. From command line:\n"
            f"     rm -rf {tensors_dir}/*.npy\n"
            f"{'=' * 70}\n"
        )
        print(error_msg, flush=True)
        raise RuntimeError(f"Debug tensors directory exceeds {SIZE_LIMIT_GB} GB limit")

    elif size_gb >= SIZE_WARNING_GB:
        tensors_dir = _get_debug_tensors_dir()
        warning_msg = (
            f"\n{'=' * 70}\n"
            f"⚠️  DEBUGGER STARTUP WARNING\n"
            f"{'=' * 70}\n"
            f"Debug tensors directory is {size_gb:.2f} GB (warning: {SIZE_WARNING_GB} GB)\n"
            f"Directory: {tensors_dir}\n"
            f"Consider cleaning up to free disk space: debug_clear()\n"
            f"{'=' * 70}\n"
        )
        print(warning_msg, flush=True)
    elif size_gb > 0:
        # Just informational if under warning threshold
        tensors_dir = _get_debug_tensors_dir()
        print(f"ℹ️  Debug tensors directory: {size_gb:.3f} GB ({tensors_dir})")

    return None


# Backward compatibility aliases (deprecated - use debug_* names instead)
swap_save = debug_save
swap_load = debug_load
swap_list = debug_list
swap_info = debug_info
swap_clear = debug_clear
swap_debug_info = debug_show_all
check_swap_directory_on_startup = check_debug_directory_on_startup
