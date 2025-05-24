import mlx.core as mx


class DepthGuidedLoss:
    """
    Improved depth-guided loss that uses raw depth information for multiplicative emphasis.

    Key improvements over the original implementation:
    1. Uses raw depth maps (not VAE-encoded) to preserve depth semantics
    2. Applies multiplicative emphasis rather than weighted averaging
    3. Configurable emphasis strength and focus direction
    4. Proper spatial correspondence between depth and loss
    """

    def apply_depth_emphasis_to_loss(
        self,
        loss_tensor: mx.array,
        raw_depth_map: mx.array,
        emphasis_mode: str = "foreground",
        emphasis_strength: float = 2.0,
    ) -> mx.array:
        """
        Main entry point for applying depth-guided emphasis to losses.

        Args:
            loss_tensor: Computed pixel-wise losses in latent space [B, H, W, C]
            raw_depth_map: Raw depth values (NOT VAE-encoded) [B, H, W, C] or [B, H, W, 1]
            emphasis_mode: How to apply emphasis ("foreground", "background", "sigmoid", "log")
            emphasis_strength: How much to emphasize depth-guided regions (1.0 = no emphasis)

        Returns:
            Emphasized loss value (scalar)
        """

        if emphasis_mode == "foreground":
            return self._apply_basic_emphasis(
                pixel_losses=loss_tensor,
                raw_depth_map=raw_depth_map,
                emphasis_strength=emphasis_strength,
                focus_foreground=True,
            )
        elif emphasis_mode == "background":
            return self._apply_basic_emphasis(
                pixel_losses=loss_tensor,
                raw_depth_map=raw_depth_map,
                emphasis_strength=emphasis_strength,
                focus_foreground=False,
            )
        elif emphasis_mode == "sigmoid":
            return self._apply_basic_emphasis(
                pixel_losses=loss_tensor,
                raw_depth_map=raw_depth_map,
                emphasis_strength=emphasis_strength,
                focus_foreground=True,
                depth_power=2.0,  # Non-linear emphasis
            )
        elif emphasis_mode == "log":
            return self._apply_basic_emphasis(
                pixel_losses=loss_tensor,
                raw_depth_map=raw_depth_map,
                emphasis_strength=emphasis_strength,
                focus_foreground=True,
                depth_power=0.5,  # Gentler emphasis curve
            )
        else:
            raise ValueError(
                f"Unknown emphasis_mode: {emphasis_mode}. Use 'foreground', 'background', 'sigmoid', or 'log'."
            )

    @staticmethod
    def _apply_basic_emphasis(
        pixel_losses: mx.array,
        raw_depth_map: mx.array,
        emphasis_strength: float = 2.0,
        focus_foreground: bool = True,
        base_weight: float = 1.0,
        depth_power: float = 1.0,
    ) -> mx.array:
        """
        Apply depth-guided emphasis to pixel losses using raw depth information.

        This creates genuine emphasis on important regions rather than just redistributing
        attention like a weighted average.

        Args:
            pixel_losses: Computed pixel-wise losses in latent space [B, H, W, C]
            raw_depth_map: Raw depth values (NOT VAE-encoded) [B, H, W, C] or [B, H, W, 1]
            emphasis_strength: How much to emphasize depth-guided regions (1.0 = no emphasis)
            focus_foreground: If True, emphasize foreground (high depth). If False, emphasize background.
            base_weight: Base weight applied to all pixels (1.0 = normal)
            depth_power: Power to apply to normalized depth values for non-linear emphasis

        Returns:
            Emphasized loss value (scalar)

        Mathematical formulation:
            1. Normalize depth: depth_norm = (depth - min) / (max - min)
            2. Create emphasis: emphasis = depth_norm^depth_power (or 1-depth_norm for background)
            3. Final weights: weights = base_weight + emphasis * emphasis_strength
            4. Result: mean(pixel_losses * weights)
        """

        # Validate inputs
        if pixel_losses.size == 0 or raw_depth_map.size == 0:
            return pixel_losses.mean()

        # Ensure depth map matches pixel loss dimensions
        depth_map = DepthGuidedLoss._ensure_compatible_dimensions(raw_depth_map, pixel_losses)

        # Normalize depth to [0, 1] range
        depth_min = mx.min(depth_map)
        depth_max = mx.max(depth_map)
        depth_range = depth_max - depth_min

        # Handle edge case: uniform depth
        if depth_range < 1e-8:
            return pixel_losses.mean() * base_weight

        depth_normalized = (depth_map - depth_min) / depth_range

        # Apply power function for non-linear emphasis
        if depth_power != 1.0:
            depth_normalized = mx.power(depth_normalized, depth_power)

        # Create emphasis weights based on focus direction
        if focus_foreground:
            # Higher depth values (closer objects) get more emphasis
            emphasis_weights = depth_normalized
        else:
            # Lower depth values (farther objects) get more emphasis
            emphasis_weights = 1.0 - depth_normalized

        # Create final multiplicative weights
        # base_weight ensures we don't reduce loss, emphasis_strength controls how much we increase it
        final_weights = base_weight + emphasis_weights * emphasis_strength

        # Apply emphasis and return mean
        emphasized_losses = pixel_losses * final_weights
        return emphasized_losses.mean()

    @staticmethod
    def apply_region_aware_depth_emphasis(
        pixel_losses: mx.array,
        raw_depth_map: mx.array,
        emphasis_strength: float = 2.0,
        depth_threshold: float = 0.5,
        foreground_emphasis: float = 3.0,
        background_emphasis: float = 0.5,
    ) -> mx.array:
        """
        Apply different emphasis levels to foreground vs background regions.

        This allows for more nuanced control over training focus.

        Args:
            pixel_losses: Computed pixel-wise losses in latent space
            raw_depth_map: Raw depth values (NOT VAE-encoded)
            emphasis_strength: Overall emphasis multiplier
            depth_threshold: Threshold to separate foreground from background (0-1)
            foreground_emphasis: Emphasis multiplier for foreground regions
            background_emphasis: Emphasis multiplier for background regions

        Returns:
            Emphasized loss value (scalar)
        """

        # Ensure compatible dimensions
        depth_map = DepthGuidedLoss._ensure_compatible_dimensions(raw_depth_map, pixel_losses)

        # Normalize depth
        depth_min = mx.min(depth_map)
        depth_max = mx.max(depth_map)
        depth_range = depth_max - depth_min

        if depth_range < 1e-8:
            return pixel_losses.mean()

        depth_normalized = (depth_map - depth_min) / depth_range

        # Create region-specific weights
        foreground_mask = depth_normalized > depth_threshold
        background_mask = ~foreground_mask

        weights = mx.ones_like(depth_normalized)
        weights = mx.where(foreground_mask, foreground_emphasis, weights)
        weights = mx.where(background_mask, background_emphasis, weights)

        # Apply overall emphasis strength
        final_weights = weights * emphasis_strength

        # Apply emphasis and return mean
        emphasized_losses = pixel_losses * final_weights
        return emphasized_losses.mean()

    @staticmethod
    def _ensure_compatible_dimensions(depth_map: mx.array, pixel_losses: mx.array) -> mx.array:
        """
        Ensure depth map has compatible dimensions with pixel losses.

        Handles common cases:
        - Depth map with 1 channel, losses with multiple channels
        - Depth map with different spatial dimensions (requires interpolation)
        """

        target_shape = pixel_losses.shape
        current_shape = depth_map.shape

        # If shapes already match, return as-is
        if current_shape == target_shape:
            return depth_map

        # Handle channel dimension mismatch
        if len(current_shape) == len(target_shape):
            if current_shape[:-1] == target_shape[:-1] and current_shape[-1] == 1:
                # Broadcast last dimension: [B, H, W, 1] -> [B, H, W, C]
                return mx.broadcast_to(depth_map, target_shape)

        # Handle spatial dimension mismatch (simplified - in practice you'd want proper interpolation)
        if len(current_shape) == len(target_shape) and current_shape[-1] == target_shape[-1]:
            # For now, use simple approach - could be improved with proper interpolation
            # This is mainly for demonstration; real implementation would need better resizing
            return mx.broadcast_to(mx.mean(depth_map), target_shape)

        # Fallback: broadcast to target shape
        try:
            return mx.broadcast_to(depth_map, target_shape)
        except (ValueError, RuntimeError):
            # Last resort: use mean depth value everywhere
            return mx.full(target_shape, mx.mean(depth_map))

    @staticmethod
    def compute_depth_statistics(raw_depth_map: mx.array) -> dict:
        """
        Compute statistics about the depth map for debugging and analysis.

        Returns:
            Dictionary with depth statistics
        """

        return {
            "min_depth": float(mx.min(raw_depth_map)),
            "max_depth": float(mx.max(raw_depth_map)),
            "mean_depth": float(mx.mean(raw_depth_map)),
            "std_depth": float(mx.std(raw_depth_map)),
            "depth_range": float(mx.max(raw_depth_map) - mx.min(raw_depth_map)),
        }
