"""Computes and persists per-target flight statistics.

Purpose:
    Implements `GroundTruthStatistics`, which derives summary flight
    metrics (speed, altitude, distance, duration) per aircraft from the
    merged ground-truth dataset, and writes them to `statistics.csv`.

Inputs:
    A `GroundTruthInspector` bound to the merged ground-truth DataFrame.

Outputs:
    A `pandas.DataFrame` (one row per TargetID) with the computed
    metrics, and a `statistics.csv` file when `save()` is called.

Engineering explanation:
    Speed is derived from the recorded velocity components
    (sqrt(VX^2 + VY^2 + VZ^2)) rather than differentiating position, since
    velocity is already directly recorded in the ground truth and is
    more numerically stable than a finite-difference estimate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .inspector import GroundTruthInspector


class GroundTruthStatistics:
    """Computes per-aircraft summary statistics from the ground-truth dataset.

    Purpose:
        Provide a single reusable component that turns the merged
        ground-truth table into per-aircraft summary metrics for
        reporting and later fusion-quality baselining.

    Inputs:
        `inspector`: a `GroundTruthInspector` bound to the merged
        ground-truth DataFrame.

    Outputs:
        `compute()` returns a DataFrame with one row per TargetID and
        columns: MaxSpeed, AvgSpeed, MaxAltitude, MinAltitude,
        TotalDistance, FlightDuration.
        `save()` writes that DataFrame to a CSV file.

    Engineering explanation:
        Altitude is read directly from the Z column, consistent with the
        Cartesian frame already used throughout the ground-truth schema;
        no coordinate transformation is performed (out of scope for
        Phase 1).
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
            df = self.inspector.get_target(target_id)
            speed = np.sqrt(df["VX"] ** 2 + df["VY"] ** 2 + df["VZ"] ** 2)
            rows.append(
                {
                    "TargetID": target_id,
                    "MaxSpeed": float(speed.max()),
                    "AvgSpeed": float(speed.mean()),
                    "MaxAltitude": float(df["Z"].max()),
                    "MinAltitude": float(df["Z"].min()),
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
