"""
Core pruning logic for removing unused files from transformers/diffusers repos.

This module provides functions to analyze execution profiles and prune unused files,
with safeguards to preserve essential infrastructure files.
"""

import json
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def categorize_file(rel_path: str) -> tuple[str, str]:
    """Categorize a file and return (category, description)."""
    if "modeling_" in rel_path and "/models/" in rel_path:
        return ("🧮 TEXT ENCODER", "Math - port to MLX")
    if rel_path.startswith("models/transformers/transformer_"):
        return ("🧮 DIFFUSION TRANSFORMER", "Math - port to MLX")
    if rel_path.startswith("models/autoencoders/"):
        return ("🧮 VAE/AUTOENCODER", "Math - port to MLX")
    if "attention" in rel_path:
        return ("🧮 ATTENTION", "Math - port to MLX")
    if "embeddings.py" in rel_path or "normalization.py" in rel_path or "activations.py" in rel_path:
        return ("🧮 BUILDING BLOCKS", "Math - port to MLX")
    if "pipeline_" in rel_path and "/pipelines/" in rel_path:
        return ("🔗 PIPELINE", "Glue - understand flow")
    if "configuration_" in rel_path:
        return ("🔗 CONFIG", "Glue - hyperparameters")
    if "processing" in rel_path or "processor" in rel_path:
        return ("🖼️  I/O", "Preprocessing")
    if "scheduler" in rel_path or "scheduling" in rel_path:
        return ("⏱️  SCHEDULER", "Denoising schedule")

    return ("🛠️  UTILS", "Utilities")


def analyze_profile(profile_path: Path) -> dict[str, Any]:
    """Analyze profile and return structured data."""
    with open(profile_path) as f:
        data = json.load(f)

    # Count calls per file (store both relative and absolute paths)
    transformers_files = defaultdict(int)
    diffusers_files = defaultdict(int)
    transformers_abs_paths: dict[str, str] = {}  # Map rel_path -> abs_path
    diffusers_abs_paths: dict[str, str] = {}  # Map rel_path -> abs_path

    for call in data["calls"]:
        file_path = call["file"]

        if "/transformers/" in file_path and "/src/transformers/" in file_path:
            rel_path = file_path.split("/src/transformers/")[1]
            transformers_files[rel_path] += 1
            if rel_path not in transformers_abs_paths:
                transformers_abs_paths[rel_path] = file_path
        elif "/diffusers/" in file_path and "/src/diffusers/" in file_path:
            rel_path = file_path.split("/src/diffusers/")[1]
            diffusers_files[rel_path] += 1
            if rel_path not in diffusers_abs_paths:
                diffusers_abs_paths[rel_path] = file_path

    # Categorize files
    categorized = {"transformers": defaultdict(list), "diffusers": defaultdict(list)}

    for file, count in transformers_files.items():
        category, desc = categorize_file(file)
        abs_path = transformers_abs_paths[file]
        categorized["transformers"][category].append((file, abs_path, count))

    for file, count in diffusers_files.items():
        category, desc = categorize_file(file)
        abs_path = diffusers_abs_paths[file]
        categorized["diffusers"][category].append((file, abs_path, count))

    # Sort by call count within each category
    for repo in categorized.values():
        for category in repo:
            repo[category].sort(key=lambda x: x[2], reverse=True)  # Sort by count (now index 2)

    return {
        "total_calls": len(data["calls"]),
        "transformers_files": transformers_files,
        "diffusers_files": diffusers_files,
        "categorized": categorized,
        "profile_data": data,
    }


def generate_markdown_report(analysis: dict[str, Any], output_path: Path, script_path: Path) -> Path:
    """Generate a markdown report of the analysis."""
    report = []
    report.append("# Execution Profile Report")
    report.append("")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Profiled script:** `{script_path.absolute()}`")

    # Get git commit if available
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            timeout=2,
        )
        if git_commit.returncode == 0:
            commit_hash = git_commit.stdout.strip()[:8]
            report.append(f"**Git commit:** `{commit_hash}`")
    except (OSError, subprocess.TimeoutExpired, subprocess.SubprocessError):
        pass  # Git not available or not a git repo

    report.append("")
    report.append(f"**Total function calls:** {analysis['total_calls']:,}")
    report.append("")
    report.append("## Summary")
    report.append("")
    report.append(f"- **Transformers files**: {len(analysis['transformers_files'])}")
    report.append(f"- **Diffusers files**: {len(analysis['diffusers_files'])}")
    report.append(f"- **Total unique files**: {len(analysis['transformers_files']) + len(analysis['diffusers_files'])}")
    report.append("")

    # Add categorized sections
    for repo_name, repo_key in [("Transformers", "transformers"), ("Diffusers", "diffusers")]:
        report.append(f"## {repo_name}")
        report.append("")

        categorized = analysis["categorized"][repo_key]

        # Sort categories by importance
        category_order = [
            "🧮 TEXT ENCODER",
            "🧮 DIFFUSION TRANSFORMER",
            "🧮 VAE/AUTOENCODER",
            "🧮 ATTENTION",
            "🧮 BUILDING BLOCKS",
            "🔗 PIPELINE",
            "🔗 CONFIG",
            "🖼️  I/O",
            "⏱️  SCHEDULER",
            "🛠️  UTILS",
        ]

        for category in category_order:
            if category not in categorized or not categorized[category]:
                continue

            files = categorized[category]
            _, desc = categorize_file(files[0][0])  # Get description from first file (rel_path)

            report.append(f"### {category}")
            report.append(f"*{desc}*")
            report.append("")
            report.append(f"**{len(files)} files (ranked by usage)**")
            report.append("")

            for rank, (rel_path, abs_path, count) in enumerate(files, 1):
                marker = "🔥" if count > 100 else "  "
                report.append(f"{rank}. {marker} `{abs_path}` — **{count:,} calls**")

            report.append("")

    report.append("---")
    report.append("")
    report.append("## Legend")
    report.append("")
    report.append("- 🔥 = Heavily used (>100 calls) - definitely needed")
    report.append("- 🧮 = Math categories - code to port to MLX")
    report.append("- 🔗 = Glue categories - understand how it connects")
    report.append("- 🖼️ 🛠️ ⏱️  = Support categories - preprocessing, utils, scheduling")

    # Write report
    with open(output_path, "w") as f:
        f.write("\n".join(report))

    return output_path


def prune_files(
    analysis: dict[str, Any],
    transformers_path: Path,
    diffusers_path: Path,
    script_name: str,
) -> tuple[int, int, int]:
    """
    Delete files that are NOT in the execution profile.

    Args:
        analysis: Analysis dict from analyze_profile()
        transformers_path: Path to transformers/src/transformers
        diffusers_path: Path to diffusers/src/diffusers
        script_name: Name of script being pruned (for commit message)

    Returns:
        Tuple of (kept_count, deleted_count, essential_kept)
    """
    # Essential files that should NEVER be deleted (even if not executed)
    # These are critical infrastructure files needed for package imports
    ESSENTIAL_PATTERNS = [
        "__init__.py",  # Package initialization
        "constants.py",  # Constants imported by __init__
        "dependency_versions_check.py",  # Version checking
        "dependency_versions_table.py",  # Version table
        "_fast.py",  # Fast implementations (often imported conditionally)
        "_base.py",  # Base classes (imported but not executed)
    ]

    # Essential root-level files (imported by utils/__init__.py or other infrastructure)
    ESSENTIAL_ROOT_FILES = [
        "video_processor.py",
        "image_processor.py",
        "optimization.py",
        "training_utils.py",
        "callbacks.py",
        "model_debugging_utils.py",  # transformers
        "modeling_utils.py",  # both
        "configuration_utils.py",  # both
        "modeling_gguf_pytorch_utils.py",  # transformers (GGUF loading)
        "convert_slow_tokenizer.py",  # transformers (tokenizer conversion)
        "audio_utils.py",  # transformers (audio processing)
        "video_utils.py",  # transformers (video metadata)
        "pytorch_utils.py",  # transformers (PyTorch utilities)
        "modeling_flash_attention_utils.py",  # transformers (flash attention)
        "safetensors_conversion.py",  # transformers (safetensors)
        "image_processing_utils.py",  # transformers (image processing)
        "image_processing_base.py",  # transformers (image processing base classes - needed by image_processing_utils)
        "image_utils.py",  # transformers (image utilities - imported but not executed)
        "image_transforms.py",  # transformers (image transforms - imported but not executed)
        "processing_utils.py",  # transformers (processing base)
        "feature_extraction_utils.py",  # transformers (feature extraction base - imported but not executed)
        "tokenization_utils.py",  # transformers (tokenization utilities - imported by tokenizers)
        "activations.py",  # transformers (activation functions - imported by model files)
        "cache_utils.py",  # transformers (cache utilities - imported by model files)
        "masking_utils.py",  # transformers (masking utilities - imported by model files)
        "modeling_layers.py",  # transformers (modeling layer utilities - imported by model files)
        "modeling_rope_utils.py",  # transformers (RoPE utilities - imported by model files)
        "modeling_outputs.py",  # transformers (model output classes - imported but not executed)
        "models/auto/modeling_auto.py",  # transformers (AutoModel classes - imported but not executed)
        "models/auto/auto_factory.py",  # transformers (AutoModel factory - imported but not executed)
        "models/auto/configuration_auto.py",  # transformers (AutoConfig classes - imported but not executed)
        "models/auto/processing_auto.py",  # transformers (AutoProcessor classes - needed by auto_docstring)
        "models/auto/feature_extraction_auto.py",  # transformers (AutoFeatureExtractor - needed by auto_docstring)
        "models/auto/image_processing_auto.py",  # transformers (AutoImageProcessor - needed by auto_docstring)
        "models/auto/tokenization_auto.py",  # transformers (AutoTokenizer - needed by auto_docstring)
        "models/auto/video_processing_auto.py",  # transformers (AutoVideoProcessor - needed by auto_docstring)
        "utils/auto_docstring.py",  # transformers (auto docstring generation - used at module level)
    ]

    # Essential directories - keep ALL files in these dirs
    # These have __init__.py files that import many modules
    ESSENTIAL_DIRS = [
        "utils/",  # All utils are infrastructure
        "integrations/",  # Integration modules (flash_attention, hub_kernels, etc.)
        "quantizers/",  # Quantizers are imported dynamically
        "loaders/",  # Model loaders
        "schedulers/",  # Scheduler algorithms
        "hooks/",  # Hook utilities
        "guiders/",  # Guidance utilities
        "loss/",  # Loss functions (transformers)
        "generation/",  # Generation utilities (GenerationMixin, utils, etc.)
        "models/unets/",  # UNet models
        "models/controlnets/",  # ControlNet models
        "models/autoencoders/",  # Autoencoder models
        "models/transformers/",  # Transformer models
        "pipelines/qwenimage/",  # QwenImage pipeline (keep all files in this pipeline)
    ]

    # Get set of files that WERE executed (define before is_essential uses it)
    executed_files = {
        "transformers": set(analysis["transformers_files"].keys()),
        "diffusers": set(analysis["diffusers_files"].keys()),
    }

    def is_essential(rel_path: str) -> bool:
        """Check if file matches any essential pattern or is in essential dir."""
        # Check if in essential directory
        for dir_pattern in ESSENTIAL_DIRS:
            if rel_path.startswith(dir_pattern) or f"/{dir_pattern}" in rel_path:
                return True

        # Check if matches essential root file
        for root_file in ESSENTIAL_ROOT_FILES:
            if rel_path == root_file or rel_path.endswith(f"/{root_file}"):
                return True

        # Check if matches essential file pattern
        for pattern in ESSENTIAL_PATTERNS:
            if rel_path.endswith(pattern) or f"/{pattern}" in rel_path:
                return True

        # Keep model files that are in the execution profile (they're needed even if not directly executed)
        # This handles cases where the script fails before these files can be executed
        if "modeling_" in rel_path and rel_path in executed_files.get("transformers", set()):
            return True
        if "pipeline_" in rel_path and rel_path in executed_files.get("diffusers", set()):
            return True

        return False

    repos: dict[str, Path] = {}
    if transformers_path.exists():
        repos["transformers"] = transformers_path
    if diffusers_path.exists():
        repos["diffusers"] = diffusers_path

    if not repos:
        print("   ⚠️  No valid repo paths found")
        return 0, 0, 0

    deleted_count = 0
    kept_count = 0
    essential_kept = 0
    files_to_delete: list[Path] = []

    for repo_name, repo_path in repos.items():
        # Get all Python files
        all_files = list(repo_path.rglob("*.py"))
        executed = executed_files[repo_name]

        # Build set of directories that have executed files
        executed_dirs = set()
        for exec_file in executed:
            # Add all parent directories
            parts = Path(exec_file).parts
            for i in range(1, len(parts)):
                executed_dirs.add("/".join(parts[:i]))

        for file_path in all_files:
            rel_path = str(file_path.relative_to(repo_path))
            file_dir = str(Path(rel_path).parent) if "/" in rel_path else ""

            if rel_path in executed:
                kept_count += 1
            elif is_essential(rel_path):
                # Keep essential files even if not executed
                kept_count += 1
                essential_kept += 1
            elif file_dir in executed_dirs:
                # Keep files in directories where at least one file was executed
                # This prevents breaking __init__.py imports
                kept_count += 1
                essential_kept += 1
            else:
                deleted_count += 1
                files_to_delete.append(file_path)

    # Delete files
    for file_path in files_to_delete:
        # Determine which repo this file belongs to
        if transformers_path and transformers_path in file_path.parents:
            rel_path = str(file_path.relative_to(transformers_path))
            repo_name = "transformers"
        elif diffusers_path and diffusers_path in file_path.parents:
            rel_path = str(file_path.relative_to(diffusers_path))
            repo_name = "diffusers"
        else:
            rel_path = str(file_path.name)
            repo_name = "unknown"

        file_path.unlink()
        print(f"   ❌ [{repo_name}] {rel_path}")

    return kept_count, deleted_count, essential_kept
