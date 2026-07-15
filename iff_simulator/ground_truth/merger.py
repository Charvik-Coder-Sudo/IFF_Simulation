"""Merges every validated per-aircraft trajectory into one ground-truth table.

Purpose:
    Implements `GroundTruthMerger`, which concatenates the per-TargetID
    DataFrames produced by `GroundTruthLoader` (and checked by
    `GroundTruthValidator`) into a single, sorted DataFrame and writes it
    to `ground_truth.csv`.

Inputs:
    `dict[str, pandas.DataFrame]` keyed by TargetID, each shaped exactly
    like `iff_simulator.ground_truth.models.REQUIRED_COLUMNS`.

Outputs:
    A single merged `pandas.DataFrame`, sorted by (Time, TargetID), and
    optionally written to a CSV file.

Engineering explanation:
    A single flat, time-sorted table is the shape every future module
    (Ownship, Geometry, Airborne PSR, IFF, Scheduler, Receiver, Decoder)
    needs: it lets a scan-time simulation step through `Time` once and
    see every aircraft present at that instant, rather than re-joining
    per-aircraft tables at each step.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .models import REQUIRED_COLUMNS


class GroundTruthMerger:
    """Merges per-aircraft trajectories into one sorted ground-truth table.

    Purpose:
        Combine independently-loaded aircraft trajectories into the
        single canonical ground-truth dataset used by every other module
        in this subsystem, and persist it as `ground_truth.csv`.

    Inputs:
        `trajectories`: dict mapping TargetID -> per-aircraft DataFrame.

    Outputs:
        `merge()` returns the combined, sorted DataFrame.
        `save()` writes that DataFrame to a CSV file at a given path.

    Engineering explanation:
        Sorting by (Time, TargetID) makes the merged table directly
        usable as a scan-by-scan feed: iterating rows in order naturally
        groups all aircraft present at the same Time together.
    """

    def __init__(self, trajectories: dict[str, pd.DataFrame]) -> None:
        self.trajectories = trajectories

    def merge(self) -> pd.DataFrame:
        """Concatenate and sort all trajectories into one DataFrame."""
        if not self.trajectories:
            raise ValueError("No trajectories to merge.")

        merged = pd.concat(self.trajectories.values(), ignore_index=True)
        merged = merged[REQUIRED_COLUMNS]
        merged = merged.sort_values(by=["Time", "TargetID"]).reset_index(drop=True)
        return merged

    def save(self, merged: pd.DataFrame, output_path: Path | str) -> Path:
        """Write the merged DataFrame to a CSV file.

        Args:
            merged: DataFrame produced by `merge()`.
            output_path: destination CSV path. Parent directories are
                created automatically.

        Returns:
            The resolved output path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_path, index=False)
        return output_path
