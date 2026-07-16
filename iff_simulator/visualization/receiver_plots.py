"""Diagnostic matplotlib visualizations of Phase 9 receiver-effects output.

Purpose:
    Implements `ReceiverEffectsPlotter`, which renders the seven
    diagnostic plots Part 10 asks for from a run's already-collected
    `DecodedIFFMeasurement`s, `ReceiverStatisticsCollector`, and (if
    logged) `IFFTrack` history -- never re-running the simulation or
    reaching back into Ground Truth.

Inputs:
    A list of `DecodedIFFMeasurement` (the run's real + false-alarm +
    fruited measurements), a `ReceiverStatisticsCollector`, and
    optionally a chronological list of `IFFTrack` snapshots (e.g. from
    `IFFTrackManager(enable_logging=True).log`).

Outputs:
    PNG files written to a given output directory.

Engineering explanation:
    Mirrors `trajectory_plot.TrajectoryPlotter`'s conventions exactly:
    the non-interactive "Agg" backend, one `_new_figure`/`_save` helper
    pair, one method per plot, and a `plot_all()` convenience method.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

from ..sensors.iff.detection import compute_pd  # noqa: E402
from ..sensors.iff.measurement import DecodedIFFMeasurement  # noqa: E402
from ..sensors.iff.receiver_statistics import ReceiverStatisticsCollector  # noqa: E402
from ..sensors.iff.track import IFFTrack, TrackStatus  # noqa: E402

_TRACK_STATUS_Y = {TrackStatus.TENTATIVE: 1, TrackStatus.CONFIRMED: 2, TrackStatus.LOST: 3}


class ReceiverEffectsPlotter:
    """Renders the Phase 9 Part 10 diagnostic plots.

    Inputs:
        measurements: every `DecodedIFFMeasurement` produced this run
            (real + false-alarm + fruited).
        statistics: the run's `ReceiverStatisticsCollector`.
        track_log: chronological `IFFTrack` snapshots, if available
            (empty list if track-manager logging was not enabled).
    """

    def __init__(
        self,
        measurements: list[DecodedIFFMeasurement],
        statistics: ReceiverStatisticsCollector,
        track_log: list[IFFTrack] | None = None,
    ) -> None:
        self.measurements = measurements
        self.statistics = statistics
        self.track_log = track_log or []

    def _new_figure(self, title: str, xlabel: str, ylabel: str):
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.4)
        return fig, ax

    def _save(self, fig, output_dir: Path, filename: str) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_detection_probability_vs_range(
        self, output_dir: Path | str, pd_model: str, pd_params: dict, max_range_m: float
    ) -> Path:
        """Analytic Pd(range) curve for the configured model."""
        fig, ax = self._new_figure("Detection Probability vs Range", "Range (m)", "Pd")
        steps = 200
        ranges = [max_range_m * i / (steps - 1) for i in range(steps)]
        pds = [compute_pd(r, pd_model, pd_params) for r in ranges]
        ax.plot(ranges, pds)
        ax.set_ylim(-0.05, 1.05)
        return self._save(fig, Path(output_dir), "detection_probability_vs_range.png")

    def plot_signal_strength_vs_range(self, output_dir: Path | str) -> Path:
        """Scatter of decoded signal strength vs range for every
        measurement that has both (VALID/FRUITED origin replies)."""
        fig, ax = self._new_figure("Signal Strength vs Range", "Range (m)", "Signal Strength")
        ranges = [m.range_m for m in self.measurements if m.signal_strength is not None]
        strengths = [m.signal_strength for m in self.measurements if m.signal_strength is not None]
        ax.scatter(ranges, strengths, s=12)
        return self._save(fig, Path(output_dir), "signal_strength_vs_range.png")

    def plot_reply_delay_histogram(self, output_dir: Path | str) -> Path:
        """Histogram of total reply delay (processing + propagation), microseconds."""
        fig, ax = self._new_figure("Reply Delay Histogram", "Delay (us)", "Count")
        delays = [
            m.processing_delay + m.propagation_delay
            for m in self.measurements
            if m.processing_delay is not None and m.propagation_delay is not None
        ]
        if delays:
            ax.hist(delays, bins=min(30, max(1, len(delays))))
        return self._save(fig, Path(output_dir), "reply_delay_histogram.png")

    def plot_receiver_load_vs_time(self, output_dir: Path | str) -> Path:
        """Replies processed per tick, over time."""
        fig, ax = self._new_figure("Receiver Load vs Time", "Time", "Replies / Tick")
        times = [t for t, _ in self.statistics.load_history]
        loads = [count for _, count in self.statistics.load_history]
        ax.step(times, loads, where="post")
        return self._save(fig, Path(output_dir), "receiver_load_vs_time.png")

    def plot_track_status_timeline(self, output_dir: Path | str) -> Path:
        """Track status (Tentative/Confirmed/Lost) vs time, one line per aircraft_id."""
        fig, ax = self._new_figure("Track Status Timeline", "Time", "Track Status")
        by_aircraft: dict[str, list[IFFTrack]] = {}
        for track in self.track_log:
            by_aircraft.setdefault(track.aircraft_id, []).append(track)
        for aircraft_id, snapshots in by_aircraft.items():
            times = [s.time for s in snapshots]
            ys = [_TRACK_STATUS_Y[s.track_status] for s in snapshots]
            ax.step(times, ys, where="post", label=aircraft_id)
        ax.set_yticks(list(_TRACK_STATUS_Y.values()))
        ax.set_yticklabels([status.value for status in _TRACK_STATUS_Y])
        if by_aircraft:
            ax.legend()
        return self._save(fig, Path(output_dir), "track_status_timeline.png")

    def plot_garbled_replies_timeline(self, output_dir: Path | str) -> Path:
        """Event plot of when replies were marked GARBLED."""
        fig, ax = self._new_figure("Garbled Replies Timeline", "Time", "Garbled Event")
        times = self.statistics.garbled_history
        ax.scatter(times, [1] * len(times), marker="|", s=200)
        ax.set_yticks([])
        return self._save(fig, Path(output_dir), "garbled_replies_timeline.png")

    def plot_false_replies_timeline(self, output_dir: Path | str) -> Path:
        """Event plot of when false-alarm replies were generated."""
        fig, ax = self._new_figure("False Replies Timeline", "Time", "False Reply Event")
        times = self.statistics.false_reply_history
        ax.scatter(times, [1] * len(times), marker="|", s=200, color="red")
        ax.set_yticks([])
        return self._save(fig, Path(output_dir), "false_replies_timeline.png")

    def plot_all(
        self,
        output_dir: Path | str,
        pd_model: str,
        pd_params: dict,
        max_range_m: float,
    ) -> list[Path]:
        """Render every diagnostic plot and save it to `output_dir`."""
        return [
            self.plot_detection_probability_vs_range(output_dir, pd_model, pd_params, max_range_m),
            self.plot_signal_strength_vs_range(output_dir),
            self.plot_reply_delay_histogram(output_dir),
            self.plot_receiver_load_vs_time(output_dir),
            self.plot_track_status_timeline(output_dir),
            self.plot_garbled_replies_timeline(output_dir),
            self.plot_false_replies_timeline(output_dir),
        ]
