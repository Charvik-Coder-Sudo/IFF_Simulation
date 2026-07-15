"""Computes and persists per-target flight statistics.

Purpose:
    Implements `GroundTruthStatistics`, which derives summary flight
    metrics (speed, altitude, distance, duration) per aircraft from a
    `Scenario`'s recorded `AircraftState` histories, and writes them to
    `statistics.csv`.

Inputs:
    A `GroundTruthInspector` bound to a `Scenario`.

Outputs:
    A `pandas.DataFrame` (one row per TargetID) with the computed
    metrics, and a `statistics.csv` file when `save()` is called.

Engineering explanation:
    Speed is derived from the recorded velocity components
    (sqrt(VX^2 + VY^2 + VZ^2)) rather than differentiating position, since
    velocity is already directly recorded in the ground truth and is
    more numerically stable than a finite-difference estimate. The
    per-sample speed values are reduced (max/mean) via a small internal
    pandas Series purely to reuse the same pairwise-summation algorithm
    used pre-refactor, so results stay numerically identical; the
    Series is never exposed outside this method.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .inspector import GroundTruthInspector


class GroundTruthStatistics:
    """Computes per-aircraft summary statistics from a Scenario's state histories.

    Purpose:
        Provide a single reusable component that turns a `Scenario`'s
        recorded `AircraftState` histories into per-aircraft summary
        metrics for reporting and later fusion-quality baselining.

    Inputs:
        `inspector`: a `GroundTruthInspector` bound to a `Scenario`.

    Outputs:
        `compute()` returns a DataFrame with one row per TargetID and
        columns: MaxSpeed, AvgSpeed, MaxAltitude, MinAltitude,
        TotalDistance, FlightDuration.
        `save()` writes that DataFrame to a CSV file.

    Engineering explanation:
        Altitude is read directly from each state's `position.z`,
        consistent with the Cartesian frame already used throughout the
        ground-truth schema; no coordinate transformation is performed
        (out of scope for Phase 1).
    """

    def __init__(self, inspector: GroundTruthInspector) -> None:
        self.inspector = inspector

    def compute(self) -> pd.DataFrame:
        """Compute summary statistics for every target.

        Returns:
            DataFrame with one row per TargetID.
        """
        rows = []
        for target_id in self.inspector.list_targets():
            history = self.inspector.get_target(target_id)
            velocity = pd.DataFrame(
                {
                    "VX": [state.velocity.x for state in history],
                    "VY": [state.velocity.y for state in history],
                    "VZ": [state.velocity.z for state in history],
                }
            )
            speed = np.sqrt(velocity["VX"] ** 2 + velocity["VY"] ** 2 + velocity["VZ"] ** 2)
            altitude = pd.Series([state.position.z for state in history])
            rows.append(
                {
                    "TargetID": target_id,
                    "MaxSpeed": float(speed.max()),
                    "AvgSpeed": float(speed.mean()),
                    "MaxAltitude": float(altitude.max()),
                    "MinAltitude": float(altitude.min()),
                    "TotalDistance": self.inspector.trajectory_length(target_id),
                    "FlightDuration": self.inspector.duration(target_id),
                }
            )
        return pd.DataFrame(rows)

    def save(self, statistics: pd.DataFrame, output_path: Path | str) -> Path:
        """Write the computed statistics DataFrame to a CSV file.

        Args:
            statistics: DataFrame produced by `compute()`.
            output_path: destination CSV path. Parent directories are
                created automatically.

        Returns:
            The resolved output path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        statistics.to_csv(output_path, index=False)
        return output_path
