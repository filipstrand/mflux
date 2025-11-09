#!/usr/bin/env python3
"""
Cleanup utility for MFLUX debugger resources.

This script removes everything in the mflux_debugger/ directory to start fresh.
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


def main():
    """Main cleanup function."""
    parser = argparse.ArgumentParser(
        description="Clean up MFLUX debugger resources - removes everything in mflux_debugger/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show what would be cleaned (dry run)
  mflux-debug-clean --dry-run

  # Clean everything without confirmation
  mflux-debug-clean --yes
        """,
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

    debugger_dir = get_debugger_dir()

    # Scan what will be cleaned
    print("=" * 70)
    print("MFLUX Debugger Cleanup Utility")
    print("=" * 70)
    print()

    if not debugger_dir.exists():
        print("✨ mflux_debugger/ directory doesn't exist - nothing to clean!")
        return 0

    file_count, size_mb = get_directory_info(debugger_dir)

    if file_count == 0:
        print("✨ No files to clean - everything is already clean!")
        return 0

    print("📁 mflux_debugger/:")
    print(f"   Path: {debugger_dir}")
    print(f"   Files: {file_count}")
    print(f"   Size: {size_mb:.2f} MB")
    print()

    # Dry run mode
    if args.dry_run:
        print("🔍 DRY RUN MODE - Nothing will be deleted")
        print("   Run without --dry-run to actually clean up")
        return 0

    # Confirmation
    if not args.yes:
        print("⚠️  This will permanently delete everything in mflux_debugger/")
        response = input("Continue? [y/N]: ").strip().lower()
        if response != "y":
            print("❌ Cancelled")
            return 1

    # Perform cleanup - remove everything in mflux_debugger/
    print()
    print("🧹 Cleaning up...")
    print()

    # Remove all contents but keep the directory itself
    for item in debugger_dir.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

    print()
    print("=" * 70)
    print("✨ Cleanup complete!")
    print(f"   Files removed: {file_count}")
    print(f"   Space freed: {size_mb:.2f} MB")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
