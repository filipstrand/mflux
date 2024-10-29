import matplotlib.pyplot as plt

from mflux.dreambooth.state.training_spec import TrainingSpec
from mflux.dreambooth.state.training_state import TrainingState


class Plotter:
    @staticmethod
    def update_loss_plot(training_spec: TrainingSpec, training_state: TrainingState) -> None:
        plt.style.use("bmh")

        # Create figure with 16:9 aspect ratio
        plt.figure(figsize=(16, 9), dpi=300)

        # Plot both lines and points
        stats = training_state.statistics
        plt.plot(stats.steps, stats.losses, "b-", linewidth=2, label="Validation Loss", zorder=1)  # Line
        plt.plot(stats.steps, stats.losses, "bo", markersize=6, label="_nolegend_", zorder=2)  # Points

        # Customize the plot
        plt.title("Validation Loss Over Time", fontsize=16, pad=20)
        plt.xlabel("Steps", fontsize=12)
        plt.ylabel("Loss", fontsize=12)

        # Set integer grid
        plt.grid(True, linestyle="--", alpha=0.7)
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # Dynamic x-axis limit with 20% padding
        max_x = max(stats.steps)
        padding = max_x * 0.2
        plt.xlim(0, max_x + padding)

        # Dynamic y-axis limit with 20% padding
        max_y = float(max(stats.losses))
        padding = max_y * 0.2
        plt.ylim(0, max_y + padding)

        plt.legend(fontsize=12)

        # Add margins for better visibility
        plt.margins(x=0.02)

        # Tight layout to prevent label cutoff
        plt.tight_layout()

        # Save to desktop with high PPI
        path = training_state.get_current_loss_plot_path(training_spec)
        plt.savefig(path, format="pdf", dpi=300, bbox_inches="tight")

        # Close the figure to free memory
        plt.close()
