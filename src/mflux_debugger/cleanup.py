#!/usr/bin/env python3
"""
Cleanup utility for MFLUX debugger resources.

This script removes all debugging artifacts to start fresh:
- Debug tensors (*.npy files)
- Generated debug images
- Debug server logs
- Debug trace files (*.json)
"""

import argparse
import shutil
import sys
from pathlib import Path


def get_debugger_dir() -> Path:
    """Get the mflux_debugger directory path (top-level, not in src)."""
    # Get repo root, then go to top-level mflux_debugger
    src_dir = Path(__file__).parent  # src/mflux_debugger
    repo_root = src_dir.parent.parent  # /Users/filip/Desktop/mflux
    return repo_root / "mflux_debugger"


def get_cleanup_targets() -> dict[str, Path]:
    """Get all directories that can be cleaned up."""
    debugger_dir = get_debugger_dir()
    logs_dir = debugger_dir / "logs"
    return {
        "tensors": debugger_dir / "tensors",
        "images": debugger_dir / "images",
        "debugger_logs": logs_dir / "debugger",  # Server logs
        "runs": logs_dir / "runs",  # Script execution logs + checkpoints
        "logs": logs_dir,  # All logs (debugger + runs)
    }


def get_directory_info(directory: Path) -> tuple[int, float]:
    """
    Get file count and size for a directory.

    Returns:
        Tuple of (file_count, size_in_mb)
    """
    if not directory.exists():
        return 0, 0.0

    files = list(directory.rglob("*"))
    file_count = sum(1 for f in files if f.is_file())
    total_bytes = sum(f.stat().st_size for f in files if f.is_file())
    size_mb = total_bytes / (1024 * 1024)

    return file_count, size_mb


def clean_directory(directory: Path, dry_run: bool = False) -> tuple[int, float]:
    """
    Clean a directory by removing all files.

    Args:
        directory: Path to clean
        dry_run: If True, don't actually delete anything

    Returns:
        Tuple of (files_removed, mb_freed)
    """
    if not directory.exists():
        return 0, 0.0

    file_count, size_mb = get_directory_info(directory)

    if not dry_run and file_count > 0:
        # Remove all files but keep the directory
        for item in directory.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

    return file_count, size_mb


def main():
    """Main cleanup function."""
    parser = argparse.ArgumentParser(
        description="Clean up MFLUX debugger resources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show what would be cleaned (dry run)
  mflux-debug-clean --dry-run

  # Clean only debug tensors
  mflux-debug-clean --target debug_tensors

  # Clean everything without confirmation
  mflux-debug-clean --yes

  # Clean specific targets
  mflux-debug-clean --target images --target debugger_logs --target runs
        """,
    )

    parser.add_argument(
        "--target",
        action="append",
        choices=["tensors", "images", "debugger_logs", "runs", "logs", "all"],
        help="Specific target(s) to clean (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be cleaned without actually deleting",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt",
    )

    args = parser.parse_args()

    # Determine targets
    targets = get_cleanup_targets()
    if args.target:
        if "all" in args.target:
            selected_targets = targets
        else:
            selected_targets = {k: v for k, v in targets.items() if k in args.target}
    else:
        selected_targets = targets

    # Scan what will be cleaned
    print("=" * 70)
    print("MFLUX Debugger Cleanup Utility")
    print("=" * 70)
    print()

    total_files = 0
    total_size_mb = 0.0
    target_info = {}

    for name, path in selected_targets.items():
        file_count, size_mb = get_directory_info(path)
        target_info[name] = (path, file_count, size_mb)
        total_files += file_count
        total_size_mb += size_mb

        if file_count > 0:
            print(f"📁 {name}:")
            print(f"   Path: {path}")
            print(f"   Files: {file_count}")
            print(f"   Size: {size_mb:.2f} MB")
            print()

    if total_files == 0:
        print("✨ No files to clean - everything is already clean!")
        return 0

    print(f"Total: {total_files} files, {total_size_mb:.2f} MB")
    print()

    # Dry run mode
    if args.dry_run:
        print("🔍 DRY RUN MODE - Nothing will be deleted")
        print("   Run without --dry-run to actually clean up")
        return 0

    # Confirmation
    if not args.yes:
        print("⚠️  This will permanently delete all files in the selected directories.")
        response = input("Continue? [y/N]: ").strip().lower()
        if response != "y":
            print("❌ Cancelled")
            return 1

    # Perform cleanup
    print()
    print("🧹 Cleaning up...")
    print()

    cleaned_files = 0
    cleaned_size_mb = 0.0

    for name, (path, file_count, size_mb) in target_info.items():
        if file_count > 0:
            removed_files, freed_mb = clean_directory(path, dry_run=False)
            cleaned_files += removed_files
            cleaned_size_mb += freed_mb
            print(f"✅ Cleaned {name}: {removed_files} files, {freed_mb:.2f} MB freed")

    print()
    print("=" * 70)
    print("✨ Cleanup complete!")
    print(f"   Files removed: {cleaned_files}")
    print(f"   Space freed: {cleaned_size_mb:.2f} MB")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
