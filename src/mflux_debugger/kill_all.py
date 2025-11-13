"""
Kill all MLX and PyTorch processes.

This utility kills all running Python processes that are using MLX or PyTorch,
including debug sessions, training runs, inference scripts, and any other
MLX/PyTorch-related Python processes.

Usage:
    mflux-debug-kill-all                    # Kill all MLX and PyTorch processes
    mflux-debug-kill-all --mlx-only        # Kill only MLX processes
    mflux-debug-kill-all --pytorch-only    # Kill only PyTorch processes
    mflux-debug-kill-all --dry-run         # Preview what would be killed
    mflux-debug-kill-all --force           # Force kill (SIGKILL) instead of graceful (SIGTERM)
"""

import argparse
import sys

import psutil


def is_mlx_process(proc):
    """Check if a process is using MLX."""
    try:
        cmdline = " ".join(proc.cmdline())
        # Check for MLX imports or usage
        if any(
            keyword in cmdline.lower()
            for keyword in [
                "import mlx",
                "from mlx",
                "mflux",
                "mlx.core",
                "mlx.nn",
                "mflux-debug-mlx",
            ]
        ):
            return True

        # Check open files for MLX libraries
        try:
            for file in proc.open_files():
                if "mlx" in file.path.lower():
                    return True
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass

        # Check memory maps for MLX libraries (additional check)
        # Note: memory_maps() is not available on macOS (removed in psutil 5.6.0+)
        # For debugger processes, cmdline check above is most reliable
        try:
            if hasattr(proc, "memory_maps"):
                for mmap in proc.memory_maps():
                    if "mlx" in mmap.path.lower():
                        return True
        except (psutil.AccessDenied, psutil.NoSuchProcess, AttributeError):
            pass

    except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
        pass

    return False


def is_pytorch_process(proc):
    """Check if a process is using PyTorch."""
    try:
        cmdline = " ".join(proc.cmdline())
        # Check for PyTorch imports or usage
        if any(
            keyword in cmdline.lower()
            for keyword in [
                "import torch",
                "from torch",
                "pytorch",
                "torch.nn",
                "mflux-debug-pytorch",
            ]
        ):
            return True

        # Check open files for PyTorch libraries
        try:
            for file in proc.open_files():
                if "torch" in file.path.lower() or "pytorch" in file.path.lower():
                    return True
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass

        # Check memory maps for PyTorch libraries (additional check)
        # Note: memory_maps() is not available on macOS (removed in psutil 5.6.0+)
        # For debugger processes, cmdline check above is most reliable
        try:
            if hasattr(proc, "memory_maps"):
                for mmap in proc.memory_maps():
                    path_lower = mmap.path.lower()
                    if "torch" in path_lower or "pytorch" in path_lower:
                        return True
        except (psutil.AccessDenied, psutil.NoSuchProcess, AttributeError):
            pass

    except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
        pass

    return False


def get_process_info(proc):
    """Get human-readable process information."""
    try:
        pid = proc.pid
        name = proc.name()
        cmdline = " ".join(proc.cmdline())
        # Truncate long command lines
        if len(cmdline) > 100:
            cmdline = cmdline[:97] + "..."
        return f"  • PID {pid}: {name} - {cmdline}"
    except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
        return f"  • PID {proc.pid}: <process info unavailable>"


def find_target_processes(mlx_only=False, pytorch_only=False):
    """Find all MLX and/or PyTorch processes."""
    mlx_processes = []
    pytorch_processes = []

    current_pid = psutil.Process().pid

    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            # Skip the current process
            if proc.pid == current_pid:
                continue

            # Skip non-Python processes for efficiency
            name = proc.name().lower()
            if "python" not in name and "mflux" not in name:
                continue

            # Check if MLX process
            if not pytorch_only and is_mlx_process(proc):
                mlx_processes.append(proc)

            # Check if PyTorch process (don't double-count)
            elif not mlx_only and is_pytorch_process(proc):
                pytorch_processes.append(proc)

        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue

    return mlx_processes, pytorch_processes


def kill_processes(processes, force=False, dry_run=False):
    """Kill a list of processes."""
    if not processes:
        return 0

    killed_count = 0
    failed = []

    for proc in processes:
        try:
            if dry_run:
                print(f"  [DRY RUN] Would kill: {get_process_info(proc)}")
                killed_count += 1
            else:
                if force:
                    proc.kill()  # SIGKILL
                    print(f"  ☠️  Force killed: {get_process_info(proc)}")
                else:
                    proc.terminate()  # SIGTERM
                    print(f"  ✅ Terminated: {get_process_info(proc)}")
                killed_count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:  # noqa: PERF203
            # Process can disappear between detection and kill - must catch in loop
            failed.append((proc.pid, str(e)))

    if failed:
        print(f"\n⚠️  Failed to kill {len(failed)} process(es):", file=sys.stderr)
        for pid, error in failed:
            print(f"  • PID {pid}: {error}", file=sys.stderr)

    return killed_count


def main():
    """Main entry point for mflux-debug-kill-all command."""
    parser = argparse.ArgumentParser(
        description="Kill all MLX and PyTorch processes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mflux-debug-kill-all                    # Kill all MLX and PyTorch processes
  mflux-debug-kill-all --mlx-only        # Kill only MLX processes
  mflux-debug-kill-all --pytorch-only    # Kill only PyTorch processes
  mflux-debug-kill-all --dry-run         # Preview what would be killed
  mflux-debug-kill-all --force           # Force kill (SIGKILL) instead of graceful (SIGTERM)

This command is useful when you've started multiple debugging sessions,
training runs, or inference scripts and want to quickly clean up all
running processes to get a clean slate.
        """,
    )

    parser.add_argument(
        "--mlx-only",
        action="store_true",
        help="Kill only MLX processes (leave PyTorch running)",
    )
    parser.add_argument(
        "--pytorch-only",
        action="store_true",
        help="Kill only PyTorch processes (leave MLX running)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be killed without actually killing anything",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force kill (SIGKILL) instead of graceful termination (SIGTERM)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.mlx_only and args.pytorch_only:
        print("❌ Error: Cannot specify both --mlx-only and --pytorch-only", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("MFLUX Kill-All Utility")
    print("=" * 70)
    print()

    if args.dry_run:
        print("🔍 DRY RUN MODE - Nothing will be killed\n")

    # Find target processes
    print("🔍 Scanning for processes...\n")
    mlx_processes, pytorch_processes = find_target_processes(
        mlx_only=args.mlx_only,
        pytorch_only=args.pytorch_only,
    )

    total_processes = len(mlx_processes) + len(pytorch_processes)

    if total_processes == 0:
        print("✅ No MLX or PyTorch processes found!")
        sys.exit(0)

    # Display what will be killed
    if mlx_processes:
        print(f"🎯 Found {len(mlx_processes)} MLX process(es):")
        for proc in mlx_processes:
            print(get_process_info(proc))
        print()

    if pytorch_processes:
        print(f"🔥 Found {len(pytorch_processes)} PyTorch process(es):")
        for proc in pytorch_processes:
            print(get_process_info(proc))
        print()

    # Kill processes (no confirmation needed - just kill)
    killed_count = 0

    if mlx_processes:
        print("🎯 Terminating MLX processes...")
        killed_count += kill_processes(mlx_processes, force=args.force, dry_run=args.dry_run)
        print()

    if pytorch_processes:
        print("🔥 Terminating PyTorch processes...")
        killed_count += kill_processes(pytorch_processes, force=args.force, dry_run=args.dry_run)
        print()

    # Summary
    print("=" * 70)
    if args.dry_run:
        print(f"✅ Dry run complete: Would kill {killed_count} process(es)")
        print("   Run without --dry-run to actually kill them")
    else:
        print(f"✅ Successfully terminated {killed_count} process(es)")
    print("=" * 70)


if __name__ == "__main__":
    main()
