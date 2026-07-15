"""Phase 1 entry point: build and inspect the Ground Truth subsystem.

Purpose:
    Runs the full Ground Truth pipeline end to end: load every aircraft
    recording from `Aircrafts/`, validate it, merge it into one dataset,
    save `ground_truth.csv`, render diagnostic plots, and compute
    `statistics.csv`.

Inputs:
    None (paths are derived relative to this file's location).

Outputs:
    - iff_simulator/output/ground_truth.csv
    - iff_simulator/output/statistics.csv
    - iff_simulator/output/*.png (diagnostic plots)
    - A console validation report and dataset summary.

Engineering explanation:
    This script intentionally contains no IFF, Mode S, Mode 5, tracking,
    or geometry logic — it only exercises the Ground Truth subsystem,
    which is the reusable foundation later phases will build on.
"""

from __future__ import annotations

from pathlib import Path

from iff_simulator.ground_truth import (
    GroundTruthInspector,
    GroundTruthLoader,
    GroundTruthMerger,
    GroundTruthStatistics,
    GroundTruthValidator,
)
from iff_simulator.visualization import TrajectoryPlotter

PROJECT_ROOT = Path(__file__).resolve().parent
AIRCRAFTS_DIR = PROJECT_ROOT / "Aircrafts"
OUTPUT_DIR = PROJECT_ROOT / "iff_simulator" / "output"


def main() -> None:
    loader = GroundTruthLoader(AIRCRAFTS_DIR)
    trajectories = loader.load()
    print(f"Loaded {len(trajectories)} target(s): {sorted(trajectories)}")

    validator = GroundTruthValidator(trajectories)
    validator.validate(verbose=True)

    merger = GroundTruthMerger(trajectories)
    merged = merger.merge()
    ground_truth_path = merger.save(merged, OUTPUT_DIR / "ground_truth.csv")
    print(f"\nSaved merged ground truth to {ground_truth_path}")

    inspector = GroundTruthInspector(merged)
    print("\nDataset summary:")
    print(inspector.summary())

    plotter = TrajectoryPlotter(inspector)
    plot_paths = plotter.plot_all(OUTPUT_DIR)
    print(f"\nSaved {len(plot_paths)} plot(s) to {OUTPUT_DIR}")

    stats = GroundTruthStatistics(inspector)
    stats_df = stats.compute()
    stats_path = stats.save(stats_df, OUTPUT_DIR / "statistics.csv")
    print(f"\nSaved statistics to {stats_path}")
    print(stats_df.to_string(index=False))


if __name__ == "__main__":
    main()
