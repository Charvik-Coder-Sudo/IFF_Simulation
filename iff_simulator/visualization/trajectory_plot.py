"""Static matplotlib visualizations of a Scenario's recorded state histories.

Purpose:
    Implements `TrajectoryPlotter`, which renders the required set of
    diagnostic plots (trajectory XY, trajectory XZ, speed vs time,
    altitude vs time, final aircraft positions) and saves each one as a
    PNG file in the output folder.

Inputs:
    A `GroundTruthInspector` bound to a `Scenario`.

Outputs:
    PNG files written to a given output directory.

Engineering explanation:
    Uses matplotlib's non-interactive "Agg" backend so plots can be
    generated headlessly (e.g. in CI or from a plain script run) without
    requiring a display, and closes each figure after saving to avoid
    unbounded memory growth when many targets are plotted. Data is read
    directly from `AircraftState` domain objects into plain Python
    lists — matplotlib needs no pandas/numpy container to plot, so no
    DataFrame appears anywhere in this module.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

from ..ground_truth.inspector import GroundTruthInspector  # noqa: E402


class TrajectoryPlotter:
    """Renders diagnostic plots of a Scenario's recorded state histories.

    Purpose:
        Provide a single reusable component that turns a `Scenario`'s
        recorded `AircraftState` histories into the standard set of
        diagnostic plots used to visually sanity-check recorded
        trajectories.

    Inputs:
        `inspector`: a `GroundTruthInspector` bound to a `Scenario`.

    Outputs:
        PNG files saved into a given output directory, one per plot.

    Engineering explanation:
        Each aircraft is plotted with a distinct color/label so multiple
        targets can be visually compared on the same axes, which is the
        most common way engineers spot-check simulated ground truth.
    """

    def __init__(self, inspector: GroundTruthInspector) -> None:
        self.inspector = inspector

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

    def plot_trajectory_xy(self, output_dir: Path | str) -> Path:
        """Plot X vs Y ground track for every target."""
        fig, ax = self._new_figure("Trajectory (XY)", "X (m)", "Y (m)")
        for target_id in self.inspector.list_targets():
            history = self.inspector.get_target(target_id)
            xs = [state.position.x for state in history]
            ys = [state.position.y for state in history]
            ax.plot(xs, ys, label=target_id)
        ax.legend()
        return self._save(fig, Path(output_dir), "trajectory_xy.png")

    def plot_trajectory_xz(self, output_dir: Path | str) -> Path:
        """Plot X vs Z (altitude profile) for every target."""
        fig, ax = self._new_figure("Trajectory (XZ)", "X (m)", "Z (m)")
        for target_id in self.inspector.list_targets():
            history = self.inspector.get_target(target_id)
            xs = [state.position.x for state in history]
            zs = [state.position.z for state in history]
            ax.plot(xs, zs, label=target_id)
        ax.legend()
        return self._save(fig, Path(output_dir), "trajectory_xz.png")

    def plot_speed_vs_time(self, output_dir: Path | str) -> Path:
        """Plot speed magnitude vs time for every target."""
        fig, ax = self._new_figure("Speed vs Time", "Time", "Speed (m/s)")
        for target_id in self.inspector.list_targets():
            history = self.inspector.get_target(target_id)
            times = [state.time for state in history]
            speeds = [state.velocity.magnitude() for state in history]
            ax.plot(times, speeds, label=target_id)
        ax.legend()
        return self._save(fig, Path(output_dir), "speed_vs_time.png")

    def plot_altitude_vs_time(self, output_dir: Path | str) -> Path:
        """Plot altitude (Z) vs time for every target."""
        fig, ax = self._new_figure("Altitude vs Time", "Time", "Z (m)")
        for target_id in self.inspector.list_targets():
            history = self.inspector.get_target(target_id)
            times = [state.time for state in history]
            zs = [state.position.z for state in history]
            ax.plot(times, zs, label=target_id)
        ax.legend()
        return self._save(fig, Path(output_dir), "altitude_vs_time.png")

    def plot_final_positions(self, output_dir: Path | str) -> Path:
        """Plot the final (X, Y) position of every target as a scatter point."""
        fig, ax = self._new_figure("Final Aircraft Positions", "X (m)", "Y (m)")
        for target_id in self.inspector.list_targets():
            history = self.inspector.get_target(target_id)
            final_state = history[-1]
            ax.scatter(final_state.position.x, final_state.position.y, label=target_id, s=60)
        ax.legend()
        return self._save(fig, Path(output_dir), "final_positions.png")

    def plot_all(self, output_dir: Path | str) -> list[Path]:
        """Render every diagnostic plot and save it to `output_dir`.

        Returns:
            List of saved file paths.
        """
        return [
            self.plot_trajectory_xy(output_dir),
            self.plot_trajectory_xz(output_dir),
            self.plot_speed_vs_time(output_dir),
            self.plot_altitude_vs_time(output_dir),
            self.plot_final_positions(output_dir),
        ]
