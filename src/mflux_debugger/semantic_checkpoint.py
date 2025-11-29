"""
Semantic checkpoint system for model comparison debugging.

Allows marking key computation points in code with named checkpoints
that can be automatically compared across different implementations.
"""

import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# Global registry of checkpoint definitions
_CHECKPOINT_REGISTRY: Dict[str, "CheckpointDefinition"] = {}

# Flag to check if debugger is active
_DEBUGGER_ACTIVE = False

# Global A/B run identifier (must be consistent across all A/B checkpoints
# within a single process).
_AB_RUN_ID: Optional[str] = None


@dataclass
class CheckpointDefinition:
    """Definition of a semantic checkpoint for comparison."""

    name: str
    description: str = ""
    variable_mapping: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # variable_mapping structure:
    # {
    #   "mlx": {"txt": "txt_attn_output", "img": "img_attn_output"},
    #   "pytorch": {"txt": "txt_attn_output", "img": "img_attn_output"}
    # }


def register_checkpoint(
    name: str,
    description: str = "",
    mlx_vars: Optional[Dict[str, str]] = None,
    pytorch_vars: Optional[Dict[str, str]] = None,
):
    """
    Register a semantic checkpoint definition.

    Args:
        name: Unique name for this checkpoint
        description: Human-readable description of what this checkpoint represents
        mlx_vars: Mapping of semantic names to MLX variable names
        pytorch_vars: Mapping of semantic names to PyTorch variable names
    """
    variable_mapping = {}
    if mlx_vars:
        variable_mapping["mlx"] = mlx_vars
    if pytorch_vars:
        variable_mapping["pytorch"] = pytorch_vars

    checkpoint = CheckpointDefinition(
        name=name,
        description=description,
        variable_mapping=variable_mapping,
    )
    _CHECKPOINT_REGISTRY[name] = checkpoint


def get_checkpoint_definition(name: str) -> Optional[CheckpointDefinition]:
    """Get a registered checkpoint definition."""
    return _CHECKPOINT_REGISTRY.get(name)


def list_checkpoints() -> Dict[str, CheckpointDefinition]:
    """List all registered checkpoints."""
    return _CHECKPOINT_REGISTRY.copy()


def set_debugger_active(active: bool):
    """Set whether the debugger is currently active."""
    global _DEBUGGER_ACTIVE
    _DEBUGGER_ACTIVE = active


def is_debugger_active() -> bool:
    """Check if debugger is currently active."""
    return _DEBUGGER_ACTIVE


def _infer_stream_from_context(frame, variables: dict) -> Optional[str]:
    """
    Automatically infer stream identifier from context.

    Strategies:
    1. Check for 'self' and look at attribute name (img_norm1 vs txt_norm1)
    2. Examine tensor shapes - text usually has ~14-20 tokens, image has ~256+ tokens
    3. Look at variable names in the frame

    Returns:
        Stream identifier ("img", "txt") or None if cannot determine
    """
    # Strategy 1: Check if we're in a method call and look at the instance
    if "self" in frame.f_locals:
        self_obj = frame.f_locals["self"]
        # Walk up the call stack to find which norm is calling us
        caller_frame = frame.f_back
        if caller_frame and "self" in caller_frame.f_locals:
            caller_self = caller_frame.f_locals["self"]
            # Check if caller has img_norm1 or txt_norm1 attributes
            if hasattr(caller_self, "img_norm1") and hasattr(caller_self, "txt_norm1"):
                if caller_self.img_norm1 is self_obj:
                    return "img"
                elif caller_self.txt_norm1 is self_obj:
                    return "txt"

    # Strategy 2: Check tensor shapes - look for 'hidden_states' variable
    if "hidden_states" in variables:
        hidden_states = variables["hidden_states"]
        # Try to get shape
        try:
            if hasattr(hidden_states, "shape"):
                shape = hidden_states.shape
                # If it's a 3D tensor [B, S, D], check sequence length
                if len(shape) >= 2:
                    seq_len = shape[1]
                    # Text is typically 14-20 tokens, image is 256+ tokens
                    if seq_len < 50:
                        return "txt"
                    elif seq_len > 100:
                        return "img"
        except (AttributeError, IndexError, TypeError):
            pass

    # Strategy 3: Check for variable name patterns
    for var_name in variables.keys():
        if "img" in var_name.lower():
            return "img"
        elif "txt" in var_name.lower() or "text" in var_name.lower():
            return "txt"

    return None


def _auto_capture_variables(frame, metadata: Optional[Dict] = None) -> Dict:
    """
    Automatically capture relevant local variables from the caller's frame.

    Captures tensor/array variables (torch.Tensor, mlx.core.array, numpy.ndarray)
    and optionally variables mentioned in metadata.

    Args:
        frame: The caller's stack frame
        metadata: Optional metadata dict that may contain variable hints

    Returns:
        Dict of variable names to values
    """
    captured = {}
    local_vars = frame.f_locals

    # Get variable names mentioned in metadata if available
    hinted_names = set()
    if metadata:
        # Look for common patterns like "tensor_name": "x" or variables mentioned in values
        for value in metadata.values():
            if isinstance(value, str) and value in local_vars:
                hinted_names.add(value)

    # Capture variables
    for name, value in local_vars.items():
        # Skip private/internal variables
        if name.startswith("_"):
            continue

        # Skip 'self' to avoid capturing entire objects
        if name == "self":
            continue

        # Capture if mentioned in metadata
        if name in hinted_names:
            captured[name] = value
            continue

        # Capture tensor/array types
        type_name = type(value).__name__
        if any(tensor_type in type_name.lower() for tensor_type in ["tensor", "array", "ndarray"]):
            captured[name] = value

    return captured


def debug_checkpoint(
    checkpoint_name: str,
    stream: Optional[str] = None,
    context: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
    skip: bool = False,
    verified: bool = False,
    terminate_after: bool = False,
    **variables,
):
    """
    Mark a semantic checkpoint in code.

    This function ALWAYS logs checkpoint state to JSON files, whether the interactive
    debugger is running or not. This enables:
    - Automatic state capture for offline analysis
    - Comparing implementations without manual breakpoints
    - No need for separate "trace mode"

    Behavior:
    - ALWAYS logs state to JSON file (in mflux_debugger/checkpoints/<script>/)
    - When interactive debugger is active AND skip=False, also pauses for inspection
    - When skip=True, logs but doesn't pause (useful for checkpoints in loops)
    - **AUTO-CAPTURE**: If no variables are passed, automatically captures all local
      tensors/arrays and variables mentioned in metadata

    **RECOMMENDED USAGE**: Place checkpoints directly in code with skip=False (default).

    Usage:
        # Auto-capture mode - captures all tensors/arrays automatically:
        debug_checkpoint("after_input_creation",
                        metadata={"step": 1, "tensor_name": "x"})

        # Explicit variables (recommended for clarity):
        debug_checkpoint("text_embeddings_generated",
                        prompt_embeds=prompt_embeds,
                        prompt_mask=prompt_mask)

        # Logs but doesn't pause (useful for loop iterations):
        debug_checkpoint("transformer_block",
                        skip=True,  # Won't pause, but still logs
                        hidden_states=hidden_states)

        # With execution context for synchronization:
        debug_checkpoint("layernorm_input",
                        context={"block": 0, "timestep": 0},
                        hidden_states=hidden_states)

        # With metadata:
        debug_checkpoint("after_attention",
                        metadata={"layer": "img", "operation": "attention"},
                        hidden_states=hidden_states)

        # Explicit stream (optional):
        debug_checkpoint("layernorm_input",
                        stream="img",
                        hidden_states=hidden_states)

        # Mark as verified (documentation only - doesn't affect execution):
        debug_checkpoint("after_patch_embed",
                        verified=True,  # Marks that this checkpoint has been verified to match
                        hidden_states=hidden_states)

    Args:
        checkpoint_name: Name of the semantic checkpoint
        stream: Optional explicit stream identifier. If not provided, will be inferred
               from context (tensor shapes, call stack, variable names)
        context: Optional dict with execution context (block, timestep, iteration, etc.)
                 for synchronization across implementations
        metadata: Optional dict with additional metadata to log
        skip: If True, logs but doesn't pause interactive debugger (useful for loops).
              Default is False (logs AND pauses when debugger is active).
        verified: If True, marks this checkpoint as verified (documentation only).
                 When hit, will display verification status. Does not affect execution.
        terminate_after: If True, terminates execution after logging checkpoint (useful for
                        debugging - stops after capturing state). Default is False.
        **variables: Named variables to capture at this checkpoint
    """
    # Get the caller's frame
    frame = inspect.currentframe()
    if frame is None:
        return

    caller_frame = frame.f_back
    if caller_frame is None:
        return

    # Auto-capture local variables if none were explicitly provided
    if not variables:
        variables = _auto_capture_variables(caller_frame, metadata)

    # Try to auto-extract context from variables if not provided
    if context is None:
        context = _extract_context_from_variables(caller_frame, variables)

    # Infer stream if not explicitly provided
    if stream is None:
        stream = _infer_stream_from_context(caller_frame, variables)

    # Construct full checkpoint name with stream prefix if we have one
    full_checkpoint_name = f"{stream}:{checkpoint_name}" if stream else checkpoint_name

    # Add skip and verified flags to metadata for logging
    log_metadata = metadata.copy() if metadata else {}
    if skip:
        log_metadata["skip"] = True
    if verified:
        log_metadata["verified"] = True

    # ALWAYS log to JSON file (new behavior!)
    _log_checkpoint_to_json(
        checkpoint_name=full_checkpoint_name,
        frame=caller_frame,
        variables=variables,
        context=context,
        metadata=log_metadata,
    )

    # Additionally, handle interactive debugger if active and not skipped
    if _DEBUGGER_ACTIVE and not skip:
        from mflux_debugger.debugger import get_active_debugger

        debugger = get_active_debugger()
        if debugger is not None:
            debugger.handle_checkpoint(
                checkpoint_name=full_checkpoint_name,
                variables=variables,
                context=context,
                frame=caller_frame,
                verified=verified,
            )
        else:
            # Log warning if debugger should be active but isn't available
            import logging

            logging.getLogger(__name__).warning(
                f"Checkpoint '{full_checkpoint_name}' hit but no active debugger found "
                f"(DEBUGGER_ACTIVE={_DEBUGGER_ACTIVE}, skip={skip})"
            )
    elif not skip:
        # Log warning if checkpoint should break but debugger isn't active
        import logging

        logging.getLogger(__name__).warning(
            f"Checkpoint '{full_checkpoint_name}' hit but debugger not active "
            f"(DEBUGGER_ACTIVE={_DEBUGGER_ACTIVE}, skip={skip})"
        )

    # Terminate execution if requested
    # But don't terminate if we're being called from MLX code (check call stack)
    if terminate_after:
        import logging
        import sys

        # Check if we're in PyTorch code being called from MLX
        # Only skip termination if we're in PyTorch/transformers code AND being called from MLX
        # caller_frame points to the wrapper function (e.g., debug_checkpoint_pytorch_B)
        # We need to walk further up to find the actual call site in processing_utils.py
        skip_termination = False
        if caller_frame:
            # Start from caller_frame and walk up to find PyTorch code and then MLX caller
            # caller_frame is the wrapper (debug_checkpoint_pytorch_B), so we need to go up further
            current_frame = caller_frame.f_back  # This should be processing_utils.py
            max_depth = 20  # Increase depth to find MLX caller through wrapper functions
            depth = 0
            found_pytorch = False

            while current_frame and depth < max_depth:
                filename = current_frame.f_code.co_filename

                # First, find if we're in PyTorch/transformers code
                if not found_pytorch and ("transformers" in filename or "diffusers" in filename):
                    found_pytorch = True
                    # Now continue walking up to find MLX caller
                    current_frame = current_frame.f_back
                    depth += 1
                    continue

                # If we found PyTorch code, now look for MLX caller
                if found_pytorch:
                    if "mflux" in filename and "models" in filename:
                        skip_termination = True
                        break

                current_frame = current_frame.f_back
                depth += 1

        logger = logging.getLogger(__name__)
        if skip_termination:
            # Don't terminate if PyTorch checkpoint is called from MLX - let MLX B checkpoint handle termination
            logger.debug(
                f"Checkpoint '{full_checkpoint_name}' would terminate, but PyTorch code called from MLX. "
                f"Skipping termination to allow MLX B checkpoint to handle it."
            )
        else:
            # Terminate normally (either MLX code or pure PyTorch code)
            termination_msg = (
                f"\n🛑 Terminating after checkpoint '{full_checkpoint_name}' "
                f"(terminate_after=True) - This is expected behavior for B checkpoints. "
                f"State has been captured and logged."
            )
            print(termination_msg)
            logger.info(termination_msg)
            sys.exit(0)


def _log_checkpoint_to_json(
    checkpoint_name: str,
    frame: Any,
    variables: Dict,
    context: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
):
    """Log checkpoint state to JSON file."""
    try:
        from pathlib import Path

        from mflux_debugger.checkpoint_writer import get_checkpoint_writer

        # Get script name from frame
        script_path = frame.f_code.co_filename
        script_name = Path(script_path).stem

        # Get checkpoint writer, passing along the current A/B run identifier
        # (if any) so that run directories can be named consistently and older
        # runs archived.
        writer = get_checkpoint_writer(script_name, ab_run_id=_AB_RUN_ID)

        # Capture checkpoint
        writer.capture_checkpoint(
            checkpoint_name=checkpoint_name,
            frame=frame,
            variables=variables,
            context=context,
            metadata=metadata,
        )
    except Exception as e:  # noqa: BLE001
        # Don't crash the program if logging fails
        import logging

        logging.getLogger(__name__).warning(f"Failed to log checkpoint '{checkpoint_name}': {e}")


def _extract_context_from_variables(frame, variables: dict) -> Dict:
    """
    Try to automatically extract execution context from frame/variables.

    Looks for common patterns like:
    - block_idx, block_num, i, idx in local variables
    - timestep, t, step in local variables
    - Walks up call stack to find loop variables
    """
    context = {}

    # Check frame locals for common context variables
    frame_locals = frame.f_locals

    # Look for block index
    for name in ["block_idx", "block_num", "i", "idx", "block_id"]:
        if name in frame_locals and isinstance(frame_locals[name], int):
            context["block"] = frame_locals[name]
            break

    # Look for timestep
    for name in ["timestep", "t", "step", "timestep_idx"]:
        if name in frame_locals:
            val = frame_locals[name]
            if isinstance(val, (int, float)):
                context["timestep"] = int(val)
                break

    # Walk up call stack to find loop variables if not found
    if "block" not in context or "timestep" not in context:
        current_frame = frame.f_back
        depth = 0
        while current_frame and depth < 10:
            locals_dict = current_frame.f_locals

            if "block" not in context:
                for name in ["block_idx", "block_num", "i"]:
                    if name in locals_dict and isinstance(locals_dict[name], int):
                        context["block"] = locals_dict[name]
                        break

            if "timestep" not in context:
                for name in ["timestep", "t", "step"]:
                    if name in locals_dict:
                        val = locals_dict[name]
                        if isinstance(val, (int, float)):
                            context["timestep"] = int(val)
                            break

            current_frame = current_frame.f_back
            depth += 1

    return context


# === Convenience wrappers for A/B-style attention comparison ==================


def _set_or_check_ab_run_id(ab_run_id: str, checkpoint_name: str) -> None:
    """
    Set the global A/B run identifier or verify it matches a previous value.

    This ensures that all four A/B checkpoints (mlx_A/B, pytorch_A/B) for a
    given process share the same run identifier.
    """
    global _AB_RUN_ID

    if _AB_RUN_ID is None:
        _AB_RUN_ID = ab_run_id
        return

    if _AB_RUN_ID != ab_run_id:
        raise ValueError(
            f"Conflicting ab_run_id for checkpoint '{checkpoint_name}': "
            f"previously '{_AB_RUN_ID}', now '{ab_run_id}'. "
            "All A/B checkpoints in a single run must share the same ab_run_id."
        )


def _merge_ab_metadata(
    ab_run_id: str,
    checkpoint_name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Merge an explicit ab_run_id into metadata, validating consistency.
    """
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise TypeError(
            f"metadata for A/B checkpoint '{checkpoint_name}' must be a dict, got {type(metadata).__name__}"
        )

    if "ab_run_id" in metadata and metadata["ab_run_id"] != ab_run_id:
        raise ValueError(
            f"Conflicting ab_run_id in metadata for '{checkpoint_name}': {metadata['ab_run_id']} vs {ab_run_id}"
        )

    metadata["ab_run_id"] = ab_run_id
    return metadata


def debug_checkpoint_mlx_A(
    *,
    ab_run_id: str,
    stream: Optional[str] = None,
    context: Optional[Dict] = None,
    metadata: Optional[Dict[str, Any]] = None,
    skip: bool = False,
    verified: bool = False,
    terminate_after: bool = False,
    **variables: Any,
) -> None:
    """
    Specialized checkpoint for MLX attention input (checkpoint A).

    This is a thin wrapper around debug_checkpoint that always uses the
    semantic name \"mlx_A\". It exists so that A/B-style debugging has a
    consistent, discoverable entry point in user code.
    """
    _set_or_check_ab_run_id(ab_run_id, "mlx_A")
    merged_meta = _merge_ab_metadata(ab_run_id, "mlx_A", metadata)
    debug_checkpoint(
        "mlx_A",
        stream=stream,
        context=context,
        metadata=merged_meta,
        skip=skip,
        verified=verified,
        terminate_after=terminate_after,
        **variables,
    )


def debug_checkpoint_mlx_B(
    *,
    ab_run_id: str,
    stream: Optional[str] = None,
    context: Optional[Dict] = None,
    metadata: Optional[Dict[str, Any]] = None,
    skip: bool = False,
    verified: bool = False,
    terminate_after: bool = True,  # Default True for B checkpoints
    **variables: Any,
) -> None:
    """
    Specialized checkpoint for MLX attention output (checkpoint B).

    Uses the semantic name \"mlx_B\".
    """
    _set_or_check_ab_run_id(ab_run_id, "mlx_B")
    merged_meta = _merge_ab_metadata(ab_run_id, "mlx_B", metadata)
    debug_checkpoint(
        "mlx_B",
        stream=stream,
        context=context,
        metadata=merged_meta,
        skip=skip,
        verified=verified,
        terminate_after=terminate_after,
        **variables,
    )


def debug_checkpoint_pytorch_A(
    *,
    ab_run_id: str,
    stream: Optional[str] = None,
    context: Optional[Dict] = None,
    metadata: Optional[Dict[str, Any]] = None,
    skip: bool = False,
    verified: bool = False,
    terminate_after: bool = False,
    **variables: Any,
) -> None:
    """
    Specialized checkpoint for PyTorch attention input (checkpoint A).

    Uses the semantic name \"pytorch_A\".
    """
    _set_or_check_ab_run_id(ab_run_id, "pytorch_A")
    merged_meta = _merge_ab_metadata(ab_run_id, "pytorch_A", metadata)
    debug_checkpoint(
        "pytorch_A",
        stream=stream,
        context=context,
        metadata=merged_meta,
        skip=skip,
        verified=verified,
        terminate_after=terminate_after,
        **variables,
    )


def debug_checkpoint_pytorch_B(
    *,
    ab_run_id: str,
    stream: Optional[str] = None,
    context: Optional[Dict] = None,
    metadata: Optional[Dict[str, Any]] = None,
    skip: bool = False,
    verified: bool = False,
    terminate_after: bool = True,  # Default True for B checkpoints
    **variables: Any,
) -> None:
    """
    Specialized checkpoint for PyTorch attention output (checkpoint B).

    Uses the semantic name \"pytorch_B\".
    """
    _set_or_check_ab_run_id(ab_run_id, "pytorch_B")
    merged_meta = _merge_ab_metadata(ab_run_id, "pytorch_B", metadata)
    debug_checkpoint(
        "pytorch_B",
        stream=stream,
        context=context,
        metadata=merged_meta,
        skip=skip,
        verified=verified,
        terminate_after=terminate_after,
        **variables,
    )
