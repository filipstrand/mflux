import matplotlib.pyplot as plt

from mflux.models.flux.variants.dreambooth.state.training_spec import TrainingSpec
from mflux.models.flux.variants.dreambooth.state.training_state import TrainingState


class Plotter:
    @staticmethod
    def update_loss_plot(training_spec: TrainingSpec, training_state: TrainingState, target_loss: float = 0.3) -> None:
        plt.style.use("bmh")

        # Create figure with 16:9 aspect ratio
        plt.figure(figsize=(16, 9), dpi=300)

        # Plot both lines and points
        stats = training_state.statistics
        plt.plot(stats.steps, stats.losses, "b-", linewidth=2, label="Validation Loss", zorder=1)  # Line
        plt.plot(stats.steps, stats.losses, "bo", markersize=6, label="_nolegend_", zorder=2)  # Points

        # Find significant points
        max_loss = float(max(stats.losses))
        max_step = int(stats.steps[stats.losses.index(max_loss)])
        min_loss = float(min(stats.losses))
        min_step = int(stats.steps[stats.losses.index(min_loss)])

        # Calculate y-axis limits with 20% padding above max loss
        y_upper = max_loss + max_loss * 0.2
        y_limits = (0, y_upper)

        # Annotate for maximum and minimum loss
        pos = Plotter._get_annotation_position(max_step, max_loss, is_max=True, y_limits=y_limits)
        plt.annotate(
            f"Max: {max_loss:.3f}\nStep: {max_step}",
            xy=(max_step, max_loss),
            xytext=pos,
            bbox=dict(facecolor="red", alpha=0.5),
            ha="center",
            arrowprops=dict(arrowstyle="->"),
            zorder=5,
        )
        plt.plot(max_step, max_loss, "ro", markersize=8, zorder=4)

        pos = Plotter._get_annotation_position(min_step, min_loss, is_max=False, y_limits=y_limits)
        plt.annotate(
            f"Min: {min_loss:.3f}\nStep: {min_step}",
            xy=(min_step, min_loss),
            xytext=pos,
            bbox=dict(facecolor="green", alpha=0.5),
            ha="center",
            arrowprops=dict(arrowstyle="->"),
            zorder=5,
        )
        plt.plot(min_step, min_loss, "go", markersize=8, zorder=4)

        # Line for target loss
        plt.axhline(y=float(target_loss), color="red", linestyle="--", linewidth=0.5, alpha=0.7)

        # Initialize counter and sum of losses
        loss_counter = 0
        loss_sum = 0
        total_step = int(training_state.iterator.total_number_of_steps())

        for loss in stats.losses:
            loss_counter += 1
            loss_sum += loss

        avg_loss = loss_sum / loss_counter if loss_counter > 0 else 0

        legend_text = [
            f"Img Dim {training_spec.width}x{training_spec.height}",
            f"Total steps: {int(max(stats.steps))} / {total_step}",
            f"Last Loss: {float(stats.losses[-1]):.4f}",
            f"Lower Loss: {min_loss:.4f} (Step {min_step})",
            f"Higher loss: {max_loss:.4f} (Step {max_step})",
            f"Avg Loss: {avg_loss:.2f}",
            f"Ideal Goal (lower of): {target_loss}",
        ]

        plt.text(
            0.98,
            0.98,
            "\n".join(legend_text),
            transform=plt.gca().transAxes,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=8,
        )

        plt.title("Validation Loss Over Time", fontsize=16, pad=20)
        plt.xlabel("Steps", fontsize=12)
        plt.ylabel("Loss", fontsize=12)

        plt.grid(True, linestyle="--", alpha=0.7)
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        plt.subplots_adjust(right=0.85)

        # Dynamic padding for x axes
        max_x = max(stats.steps)
        initial_padding = 0.4
        final_padding = 0.01
        padding_limit = initial_padding - (initial_padding - final_padding) * (max_x / total_step)
        padding = max_x * padding_limit

        plt.xlim(0, max_x + padding)

        plt.ylim(y_limits)
        plt.legend(fontsize=12)

        plt.margins(x=0.02)

        plt.tight_layout()

        path = training_state.get_current_loss_plot_path(training_spec)
        plt.savefig(path, format="pdf", dpi=300, bbox_inches="tight")

        plt.close()

    @staticmethod
    def _get_annotation_position(x, y, is_max=True, y_limits=None):
        vertical_offset = 0.10
        horizontal_offset = 0.10
        if y_limits is None:
            y_limits = plt.ylim()
        if is_max:
            if y + vertical_offset > y_limits[1]:
                return float(x - horizontal_offset), float(y)
            return float(x), float(y + vertical_offset)
        else:
            if y - vertical_offset < y_limits[0]:
                return float(x - horizontal_offset), float(y)
            return float(x), float(y - vertical_offset)
