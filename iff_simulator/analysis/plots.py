"""Diagnostic matplotlib visualizations for Phase 10 analysis output.

Purpose:
    Implements `AnalysisPlotter`, which renders the nine plots this
    phase's spec asks for, computed from a `PipelineRunRecord` via the
    other `iff_simulator.analysis` modules -- never re-running the
    simulation or reaching into Ground Truth beyond what
    `confusion_matrix.py` already does.

Inputs:
    A `PipelineRunRecord`.

Outputs:
    PNG files written to a given output directory.

Engineering explanation:
    Mirrors `receiver_plots.ReceiverEffectsPlotter` / `trajectory_plot.
    TrajectoryPlotter`'s conventions exactly: the non-interactive "Agg"
    backend, one `_new_figure`/`_save` helper pair, one method per plot,
    and a `plot_all()` convenience method.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

from ..sensors.iff import MeasurementStatus  # noqa: E402
from ..sensors.iff.detection import compute_pd  # noqa: E402
from .confusion_matrix import compute_authentication_confusion_matrix, compute_identity_confusion_matrix  # noqa: E402
from .latency_analysis import compute_latency_breakdown  # noqa: E402
from .roc_analysis import compute_roc_curve  # noqa: E402
from .run_record import PipelineRunRecord  # noqa: E402
from .statistics import compute_track_statistics  # noqa: E402


class AnalysisPlotter:
    """Renders the Phase 10 diagnostic plots.

    Inputs:
        record: the `PipelineRunRecord` to analyze and plot.
    """

    def __init__(self, record: PipelineRunRecord) -> None:
        self.record = record

    def _new_figure(self, title: str, xlabel: str, ylabel: str):
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.4)
        return fig, ax

    def _save(self, fig, output_dir: Path, filename: str) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def _all_measurements(self):
        for tick in self.record.tick_results:
            if tick.real_measurement is not None:
                yield tick.real_measurement
            yield from tick.false_alarm_measurements
            yield from tick.fruited_measurements

    def plot_roc_curve(self, output_dir: Path | str) -> Path:
        """ROC curve for the receiver's sensitivity threshold (Part 5/Phase 9)."""
        roc = compute_roc_curve(self.record)
        fig, ax = self._new_figure("ROC Curve (Sensitivity Threshold)", "False Positive Rate", "True Positive Rate")
        ax.plot(roc.false_positive_rate, roc.true_positive_rate, marker="o", markersize=3, label="ROC")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(title=f"AUC = {roc.area_under_curve:.3f}")
        return self._save(fig, output_dir, "roc_curve.png")

    def plot_detection_probability_vs_range(
        self, output_dir: Path | str, pd_model: str, pd_params: dict, max_range_m: float
    ) -> Path:
        """Analytic Pd(range) curve for the configured model (same
        derivation `receiver_plots.py` already uses -- here for a
        self-contained analysis-package artifact set)."""
        fig, ax = self._new_figure("Detection Probability vs Range", "Range (m)", "Pd")
        steps = 200
        ranges = [max_range_m * i / (steps - 1) for i in range(steps)]
        pds = [compute_pd(r, pd_model, pd_params) for r in ranges]
        ax.plot(ranges, pds)
        ax.set_ylim(-0.05, 1.05)
        return self._save(fig, output_dir, "detection_probability_vs_range.png")

    def plot_signal_strength_vs_range(self, output_dir: Path | str) -> Path:
        fig, ax = self._new_figure("Signal Strength vs Range", "Range (m)", "Signal Strength")
        measurements = [m for m in self._all_measurements() if m.signal_strength is not None]
        ax.scatter([m.range_m for m in measurements], [m.signal_strength for m in measurements], s=12)
        return self._save(fig, output_dir, "signal_strength_vs_range.png")

    def plot_latency_histogram(self, output_dir: Path | str) -> Path:
        fig, ax = self._new_figure("Total End-to-End Latency Histogram", "Delay (us)", "Count")
        measurements = [
            m for m in (t.real_measurement for t in self.record.tick_results)
            if m is not None and m.reply_status == MeasurementStatus.VALID and m.arrival_time is not None
        ]
        delays = [(m.arrival_time - m.time) * 1_000_000.0 for m in measurements]
        if delays:
            ax.hist(delays, bins=min(30, max(1, len(delays))))
        return self._save(fig, output_dir, "latency_histogram.png")

    def plot_authentication_pie_chart(self, output_dir: Path | str) -> Path:
        matrix = compute_authentication_confusion_matrix(self.record)
        authenticated = sum(matrix.counts["AUTHENTICATED"].values())
        failed = sum(matrix.counts["FAILED"].values())
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_title("Authentication Outcomes (Mode 5)")
        values = [authenticated, failed]
        if sum(values) > 0:
            ax.pie(values, labels=["Authenticated", "Failed"], autopct="%1.1f%%")
        return self._save(fig, output_dir, "authentication_pie_chart.png")

    def plot_track_lifetime_histogram(self, output_dir: Path | str) -> Path:
        fig, ax = self._new_figure("Track Lifetime Histogram", "Duration (s)", "Count")
        durations = [s.duration for s in self.record.completed_track_summaries]
        if durations:
            ax.hist(durations, bins=min(30, max(1, len(durations))))
        return self._save(fig, output_dir, "track_lifetime_histogram.png")

    def plot_detection_heatmap(self, output_dir: Path | str) -> Path:
        """2D histogram of VALID-detection density over range/azimuth."""
        fig, ax = self._new_figure("Detection Heatmap", "Azimuth (deg)", "Range (m)")
        measurements = [
            m for m in (t.real_measurement for t in self.record.tick_results)
            if m is not None and m.reply_status == MeasurementStatus.VALID
        ]
        if measurements:
            ax.hist2d(
                [m.azimuth_deg for m in measurements], [m.range_m for m in measurements],
                bins=(20, 20),
            )
        return self._save(fig, output_dir, "detection_heatmap.png")

    def plot_confusion_matrix_heatmap(self, output_dir: Path | str) -> Path:
        matrix = compute_identity_confusion_matrix(self.record)
        fig, ax = self._new_figure("Identity Confusion Matrix", "Predicted", "Ground Truth")
        grid = [[matrix.counts[t][p] for p in matrix.labels] for t in matrix.labels]
        im = ax.imshow(grid, cmap="Blues")
        ax.set_xticks(range(len(matrix.labels)))
        ax.set_xticklabels(matrix.labels)
        ax.set_yticks(range(len(matrix.labels)))
        ax.set_yticklabels(matrix.labels)
        for i, row in enumerate(grid):
            for j, value in enumerate(row):
                ax.text(j, i, str(value), ha="center", va="center")
        fig.colorbar(im, ax=ax)
        return self._save(fig, output_dir, "confusion_matrix_heatmap.png")

    def plot_receiver_statistics_dashboard(self, output_dir: Path | str) -> Path:
        """Multi-panel summary of ReceiverStatistics for this run."""
        stats = self.record.receiver_statistics
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        fig.suptitle("Receiver Statistics Dashboard")

        counts_ax = axes[0][0]
        labels = ["Received", "Lost", "Garbled", "Fruited", "False"]
        values = (
            [stats.replies_received, stats.replies_lost, stats.replies_garbled, stats.replies_fruited, stats.false_replies]
            if stats else [0, 0, 0, 0, 0]
        )
        counts_ax.bar(labels, values)
        counts_ax.set_title("Reply Outcome Counts")
        counts_ax.tick_params(axis="x", rotation=30)

        track_ax = axes[0][1]
        track_rows = compute_track_statistics(self.record)
        status_counts: dict = {}
        for row in track_rows:
            status_counts[row.track_status] = status_counts.get(row.track_status, 0) + 1
        if status_counts:
            track_ax.bar(list(status_counts.keys()), list(status_counts.values()))
        track_ax.set_title("Track Status Counts")

        summary_ax = axes[1][0]
        summary_ax.axis("off")
        summary_text = (
            f"Avg Detection Pd: {stats.average_detection_probability:.3f}\n"
            f"Avg Signal Strength: {stats.average_signal_strength:.3f}\n"
            f"Avg Delay (us): {stats.average_delay_us:.2f}\n"
            f"Receiver Load: {stats.receiver_load:.2f}"
        ) if stats else "No receiver statistics available."
        summary_ax.text(0.05, 0.5, summary_text, fontsize=12, va="center")
        summary_ax.set_title("Run Summary")

        load_ax = axes[1][1]
        if stats and stats.replies_received + stats.false_replies + stats.replies_fruited > 0:
            load_ax.pie(
                [stats.replies_received, stats.false_replies, stats.replies_fruited],
                labels=["Real", "False", "Fruited"], autopct="%1.1f%%",
            )
        load_ax.set_title("Reply Origin Mix")

        return self._save(fig, output_dir, "receiver_statistics_dashboard.png")

    def plot_all(
        self, output_dir: Path | str, pd_model: str, pd_params: dict, max_range_m: float
    ) -> list:
        """Render every diagnostic plot and save it to `output_dir`."""
        return [
            self.plot_roc_curve(output_dir),
            self.plot_detection_probability_vs_range(output_dir, pd_model, pd_params, max_range_m),
            self.plot_signal_strength_vs_range(output_dir),
            self.plot_latency_histogram(output_dir),
            self.plot_authentication_pie_chart(output_dir),
            self.plot_track_lifetime_histogram(output_dir),
            self.plot_detection_heatmap(output_dir),
            self.plot_confusion_matrix_heatmap(output_dir),
            self.plot_receiver_statistics_dashboard(output_dir),
        ]
