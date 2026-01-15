#!/usr/bin/env python3
"""
Qwen-Image Optimization Benchmark Suite

Tests performance improvements from optimization phases:
- Phase 0: mx.eval() removal, tuple cache keys, cache limits (17-28% baseline)
- Phase 1: GPU-CPU sync, VAE transposes, RoPE (25-39% additional)
- Phase 2: Padding, attention mask (5-8% additional)
- Phase 3: Batched guidance (40-50% on transformer)
- Phase 4: Quality & memory optimizations

Expected total speedup: 1.6-2.1x (60-110% faster)
"""

import time
from pathlib import Path

import mlx.core as mx

from mflux.models.qwen.variants.txt2img import QwenImage


def benchmark_generation(
    model: QwenImage,
    size: tuple[int, int],
    steps: int,
    num_runs: int = 3,
    warmup_runs: int = 1,
) -> dict:
    """Benchmark image generation with the given parameters."""
    width, height = size
    prompt = "a serene mountain landscape at sunset, highly detailed, 8k"
    negative_prompt = "blurry, low quality, distorted"

    print(f"\n{'=' * 60}")
    print(f"Benchmarking {width}x{height} @ {steps} steps")
    print(f"{'=' * 60}")

    # Warmup runs
    print(f"Running {warmup_runs} warmup iteration(s)...")
    for i in range(warmup_runs):
        _ = model.generate_image(
            seed=42 + i,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
        )
        print(f"  Warmup {i + 1}/{warmup_runs} complete")

    # Actual benchmark runs
    print(f"\nRunning {num_runs} benchmark iteration(s)...")
    times = []
    for i in range(num_runs):
        start = time.time()
        result = model.generate_image(
            seed=100 + i,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
        )
        # Ensure computation is complete
        mx.eval(result)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {i + 1}/{num_runs}: {elapsed:.2f}s")

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print("\nResults:")
    print(f"  Average: {avg_time:.2f}s")
    print(f"  Min:     {min_time:.2f}s")
    print(f"  Max:     {max_time:.2f}s")
    print(f"  Speed:   {avg_time / steps:.3f}s per step")

    return {
        "size": f"{width}x{height}",
        "steps": steps,
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
        "time_per_step": avg_time / steps,
    }


def main():
    print("=" * 60)
    print("Qwen-Image Optimization Benchmark Suite")
    print("=" * 60)
    print(f"\nMLX Version: {mx.__version__ if hasattr(mx, '__version__') else 'N/A'}")
    print(f"Device: {mx.default_device()}")

    # Check available memory
    import subprocess

    try:
        result = subprocess.run(["sysctl", "hw.memsize"], capture_output=True, text=True, check=True)
        mem_bytes = int(result.stdout.split(":")[1].strip())
        mem_gb = mem_bytes / (1024**3)
        print(f"System RAM: {mem_gb:.1f} GB")
    except (subprocess.CalledProcessError, ValueError, IndexError):
        print("System RAM: Unknown")

    print("\nLoading model (quantized 8-bit)...")
    model = QwenImage(quantize=8)
    print("Model loaded successfully!")

    # Test configurations
    configs = [
        # (width, height, steps, runs)
        (512, 512, 20, 3),  # Small, many steps
        (1024, 1024, 20, 3),  # Standard size
        (2048, 2048, 10, 2),  # Large size (Phase 4.3 enabled)
    ]

    results = []
    for width, height, steps, runs in configs:
        result = benchmark_generation(
            model=model,
            size=(width, height),
            steps=steps,
            num_runs=runs,
        )
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"\n{'Size':<12} {'Steps':<8} {'Avg Time':<12} {'Per Step':<12}")
    print("-" * 60)
    for r in results:
        print(f"{r['size']:<12} {r['steps']:<8} {r['avg_time']:>10.2f}s  {r['time_per_step']:>10.3f}s")

    print("\n" + "=" * 60)
    print("Expected improvements (vs unoptimized baseline):")
    print("  Phase 0 (already applied):  ~1.20x (20% faster)")
    print("  Phase 1 (critical):         +25-39% additional")
    print("  Phase 2 (vectorization):    +5-8% additional")
    print("  Phase 3 (batched guidance): +40-50% additional")
    print("  Phase 4 (quality/memory):   +5-10% additional")
    print("  " + "-" * 55)
    print("  Total expected speedup:     1.6-2.1x (60-110% faster)")
    print("=" * 60)

    # Save results
    output_file = Path("benchmark_results_qwen.txt")
    with open(output_file, "w") as f:
        f.write("Qwen-Image Optimization Benchmark Results\n")
        f.write("=" * 60 + "\n\n")
        for r in results:
            f.write(f"{r['size']:<12} {r['steps']:<8} {r['avg_time']:>10.2f}s  ")
            f.write(f"{r['time_per_step']:>10.3f}s per step\n")

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
