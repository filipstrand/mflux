"""Memory optimization utilities for Z-Image training on Apple Silicon.

Designed for Mac Studio M3 Ultra with 512GB unified memory.
Provides utilities for:
- Memory estimation
- Optimal batch size calculation
- Gradient accumulation configuration
"""


class MemoryOptimizer:
    """Memory optimization for Z-Image training on Apple Silicon."""

    # Z-Image model sizes (approximate, in GB)
    MODEL_SIZE_BF16 = 12.0  # 6B params * 2 bytes
    MODEL_SIZE_8BIT = 6.0  # 6B params * 1 byte
    MODEL_SIZE_4BIT = 3.0  # 6B params * 0.5 byte

    # Per-image memory overhead (approximate, in GB)
    LATENT_OVERHEAD_PER_IMAGE = 0.05  # 1024x1024 latent
    EMBEDDING_OVERHEAD_PER_IMAGE = 0.1  # Text embeddings
    ACTIVATION_OVERHEAD_PER_IMAGE = 15.0  # Full fine-tuning
    LORA_ACTIVATION_PER_IMAGE = 3.0  # LoRA mode (much smaller)

    # Optimizer state multipliers
    ADAMW_STATE_MULTIPLIER = 2.0  # momentum + variance

    @staticmethod
    def get_available_memory() -> float:
        """Get available unified memory in GB.

        Returns system memory minus OS overhead estimate.
        """
        try:
            # Try to get actual memory info
            import subprocess

            result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
            total_bytes = int(result.stdout.strip())
            total_gb = total_bytes / (1024**3)

            # Reserve ~10% for OS and other processes
            available = total_gb * 0.90
            return available
        except Exception:  # noqa: BLE001 - Intentional: fallback for memory probing failures
            # Fallback to conservative default
            return 450.0  # Assume 512GB with overhead

    @staticmethod
    def estimate_memory_for_lora_training(
        batch_size: int,
        quantize: int | None = None,
        width: int = 1024,
        height: int = 1024,
        lora_params_millions: float = 100.0,
    ) -> dict[str, float]:
        """Estimate memory usage for LoRA training.

        Args:
            batch_size: Training batch size
            quantize: Quantization level (4, 8, or None for full precision)
            width: Image width
            height: Image height
            lora_params_millions: Estimated LoRA parameters in millions

        Returns:
            Dictionary with memory breakdown in GB
        """
        # Base model size
        if quantize == 4:
            model_size = MemoryOptimizer.MODEL_SIZE_4BIT
        elif quantize == 8:
            model_size = MemoryOptimizer.MODEL_SIZE_8BIT
        else:
            model_size = MemoryOptimizer.MODEL_SIZE_BF16

        # Scale latent overhead by resolution
        resolution_scale = (width * height) / (1024 * 1024)
        latent_overhead = MemoryOptimizer.LATENT_OVERHEAD_PER_IMAGE * resolution_scale * batch_size

        # Embeddings
        embedding_overhead = MemoryOptimizer.EMBEDDING_OVERHEAD_PER_IMAGE * batch_size

        # Activations (LoRA mode - smaller)
        activation_overhead = MemoryOptimizer.LORA_ACTIVATION_PER_IMAGE * batch_size

        # LoRA parameters and optimizer state
        lora_params_gb = (lora_params_millions * 1e6 * 2) / (1024**3)  # bf16
        optimizer_state = lora_params_gb * MemoryOptimizer.ADAMW_STATE_MULTIPLIER

        return {
            "model": model_size,
            "latents": latent_overhead,
            "embeddings": embedding_overhead,
            "activations": activation_overhead,
            "lora_params": lora_params_gb,
            "optimizer": optimizer_state,
            "total": model_size
            + latent_overhead
            + embedding_overhead
            + activation_overhead
            + lora_params_gb
            + optimizer_state,
        }

    @staticmethod
    def estimate_memory_for_full_training(
        batch_size: int,
        width: int = 1024,
        height: int = 1024,
    ) -> dict[str, float]:
        """Estimate memory usage for full fine-tuning.

        Note: Full fine-tuning requires full precision model.

        Args:
            batch_size: Training batch size
            width: Image width
            height: Image height

        Returns:
            Dictionary with memory breakdown in GB
        """
        # Full precision model required
        model_size = MemoryOptimizer.MODEL_SIZE_BF16

        # Scale by resolution
        resolution_scale = (width * height) / (1024 * 1024)
        latent_overhead = MemoryOptimizer.LATENT_OVERHEAD_PER_IMAGE * resolution_scale * batch_size

        # Embeddings
        embedding_overhead = MemoryOptimizer.EMBEDDING_OVERHEAD_PER_IMAGE * batch_size

        # Full activations
        activation_overhead = MemoryOptimizer.ACTIVATION_OVERHEAD_PER_IMAGE * batch_size

        # Gradients (same size as model)
        gradient_overhead = model_size

        # Optimizer state (2x for AdamW)
        optimizer_state = model_size * MemoryOptimizer.ADAMW_STATE_MULTIPLIER

        return {
            "model": model_size,
            "latents": latent_overhead,
            "embeddings": embedding_overhead,
            "activations": activation_overhead,
            "gradients": gradient_overhead,
            "optimizer": optimizer_state,
            "total": model_size
            + latent_overhead
            + embedding_overhead
            + activation_overhead
            + gradient_overhead
            + optimizer_state,
        }

    @staticmethod
    def calculate_optimal_batch_size(
        mode: str = "lora",
        available_memory_gb: float | None = None,
        quantize: int | None = None,
        width: int = 1024,
        height: int = 1024,
        safety_margin: float = 0.85,  # Use 85% of available
    ) -> tuple[int, dict]:
        """Calculate optimal batch size for available memory.

        Args:
            mode: "lora" or "full"
            available_memory_gb: Available memory (auto-detected if None)
            quantize: Quantization level (for LoRA mode)
            width: Image width
            height: Image height
            safety_margin: Fraction of memory to use (default 85%)

        Returns:
            (optimal_batch_size, memory_estimate)
        """
        if available_memory_gb is None:
            available_memory_gb = MemoryOptimizer.get_available_memory()

        usable_memory = available_memory_gb * safety_margin

        # Binary search for optimal batch size
        min_batch = 1
        max_batch = 64

        while min_batch < max_batch:
            mid_batch = (min_batch + max_batch + 1) // 2

            if mode == "lora":
                estimate = MemoryOptimizer.estimate_memory_for_lora_training(
                    batch_size=mid_batch,
                    quantize=quantize,
                    width=width,
                    height=height,
                )
            else:
                estimate = MemoryOptimizer.estimate_memory_for_full_training(
                    batch_size=mid_batch,
                    width=width,
                    height=height,
                )

            if estimate["total"] <= usable_memory:
                min_batch = mid_batch
            else:
                max_batch = mid_batch - 1

        # Get final estimate
        if mode == "lora":
            final_estimate = MemoryOptimizer.estimate_memory_for_lora_training(
                batch_size=min_batch,
                quantize=quantize,
                width=width,
                height=height,
            )
        else:
            final_estimate = MemoryOptimizer.estimate_memory_for_full_training(
                batch_size=min_batch,
                width=width,
                height=height,
            )

        return min_batch, final_estimate

    @staticmethod
    def suggest_gradient_accumulation(
        desired_batch_size: int,
        optimal_batch_size: int,
    ) -> int:
        """Suggest gradient accumulation steps to achieve desired effective batch size.

        Args:
            desired_batch_size: Target effective batch size
            optimal_batch_size: Maximum batch size that fits in memory

        Returns:
            Number of gradient accumulation steps
        """
        if desired_batch_size <= optimal_batch_size:
            return 1

        return (desired_batch_size + optimal_batch_size - 1) // optimal_batch_size

    @staticmethod
    def print_recommendations(
        mode: str = "lora",
        available_memory_gb: float | None = None,
        quantize: int | None = None,
    ) -> None:
        """Print memory optimization recommendations.

        Args:
            mode: "lora" or "full"
            available_memory_gb: Available memory (auto-detected if None)
            quantize: Quantization level
        """
        if available_memory_gb is None:
            available_memory_gb = MemoryOptimizer.get_available_memory()

        print(f"\n{'=' * 50}")
        print("Z-Image Training Memory Recommendations")
        print(f"{'=' * 50}")
        print(f"Mode: {mode.upper()}")
        print(f"Available Memory: {available_memory_gb:.1f} GB")
        print(f"Quantization: {quantize or 'None (full precision)'}")
        print()

        optimal_batch, estimate = MemoryOptimizer.calculate_optimal_batch_size(
            mode=mode,
            available_memory_gb=available_memory_gb,
            quantize=quantize,
        )

        print(f"Recommended batch_size: {optimal_batch}")
        print(f"Estimated memory usage: {estimate['total']:.1f} GB")
        print()
        print("Memory breakdown:")
        for key, value in estimate.items():
            if key != "total":
                print(f"  {key}: {value:.1f} GB")
        print()

        # Gradient accumulation suggestion
        if optimal_batch < 16:
            accum_steps = MemoryOptimizer.suggest_gradient_accumulation(16, optimal_batch)
            print(f"For effective batch_size=16, use gradient_accumulation_steps={accum_steps}")

        if optimal_batch < 32:
            accum_steps = MemoryOptimizer.suggest_gradient_accumulation(32, optimal_batch)
            print(f"For effective batch_size=32, use gradient_accumulation_steps={accum_steps}")

        print(f"{'=' * 50}")
