from mflux.models.z_image.variants.training.state.training_spec import TrainingSpec
from mflux.models.z_image.variants.training.state.training_state import TrainingState


class Plotter:
    """Loss plotting for training visualization."""

    @staticmethod
    def update_loss_plot(training_spec: TrainingSpec, training_state: TrainingState) -> None:
        """Update the loss plot with current training statistics."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            # Matplotlib not available, skip plotting
            return

        if len(training_state.statistics.steps) < 2:
            return

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot loss curve
        ax.plot(
            training_state.statistics.steps,
            training_state.statistics.losses,
            "b-",
            linewidth=1.5,
            label="Training Loss",
        )

        # Add smoothed curve if enough points
        if len(training_state.statistics.losses) > 10:
            smoothed = Plotter._moving_average(training_state.statistics.losses, window=10)
            ax.plot(
                training_state.statistics.steps[-len(smoothed) :],
                smoothed,
                "r-",
                linewidth=2,
                alpha=0.8,
                label="Smoothed (10-step avg)",
            )

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Loss")
        ax.set_title(f"Z-Image Training Loss - Step {training_state.iterator.num_iterations}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Save plot
        plot_path = training_state.get_current_loss_plot_path(training_spec)
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _moving_average(data: list[float], window: int) -> list[float]:
        """Compute moving average with specified window size."""
        if len(data) < window:
            return data

        result = []
        for i in range(len(data) - window + 1):
            avg = sum(data[i : i + window]) / window
            result.append(avg)
        return result
