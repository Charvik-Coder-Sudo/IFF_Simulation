"""Read-only query interface over the merged ground-truth dataset.

Purpose:
    Implements `GroundTruthInspector`, a convenience layer for asking
    common questions about the merged ground-truth table without every
    caller re-implementing the same pandas filtering/aggregation logic.

Inputs:
    The merged `pandas.DataFrame` produced by `GroundTruthMerger.merge()`.

Outputs:
    Scalars, DataFrames, and dictionaries answering targeted questions
    (which targets exist, a target's full trajectory, all aircraft at a
    given time, bounding boxes, durations, sample rates, path lengths).

Engineering explanation:
    This is a pure read-only, side-effect-free query layer: it never
    mutates the merged DataFrame, which lets multiple future modules
    (Ownship, Geometry, PSR, IFF, Scheduler, Receiver, Decoder) share one
    Inspector instance safely.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class GroundTruthInspector:
    """Answers common questions about the merged ground-truth dataset.

    Purpose:
        Provide a single, reusable, read-only query API over the merged
        ground-truth table, so future modules never need to hand-roll
        pandas filtering/aggregation for basic questions.

    Inputs:
        `ground_truth`: the merged DataFrame produced by
        `GroundTruthMerger.merge()`.

    Outputs:
        See individual method docstrings.

    Engineering explanation:
        Every method here is a pure function of the stored DataFrame —
        no internal state is mutated, so the same Inspector instance can
        be safely reused and shared across modules.
    """

    def __init__(self, ground_truth: pd.DataFrame) -> None:
        self.ground_truth = ground_truth

    def list_targets(self) -> list[str]:
        """Return the sorted list of distinct TargetIDs present."""
        return sorted(self.ground_truth["TargetID"].unique().tolist())

    def get_target(self, target_id: str) -> pd.DataFrame:
        """Return the full time-ordered trajectory for one TargetID."""
        self._require_target(target_id)
        return (
            self.ground_truth[self.ground_truth["TargetID"] == target_id]
            .sort_values("Time")
            .reset_index(drop=True)
        )

    def get_time(self, time: float) -> pd.DataFrame:
        """Return the state of every aircraft present at a given Time."""
        return (
            self.ground_truth[self.ground_truth["Time"] == time]
            .sort_values("TargetID")
            .reset_index(drop=True)
        )

    def summary(self) -> dict:
        """Return a dict summarizing the whole dataset: target count,
        row count, and time range."""
        return {
            "target_count": len(self.list_targets()),
            "targets": self.list_targets(),
            "total_rows": len(self.ground_truth),
            "time_min": float(self.ground_truth["Time"].min()),
            "time_max": float(self.ground_truth["Time"].max()),
        }

    def bounding_box(self, target_id: str | None = None) -> dict:
        """Return the min/max X, Y, Z extent, for one target or the whole dataset."""
        df = self.get_target(target_id) if target_id else self.ground_truth
        return {
            "x_min": float(df["X"].min()),
            "x_max": float(df["X"].max()),
            "y_min": float(df["Y"].min()),
            "y_max": float(df["Y"].max()),
            "z_min": float(df["Z"].min()),
            "z_max": float(df["Z"].max()),
        }

    def duration(self, target_id: str | None = None) -> float:
        """Return the flight duration (max Time - min Time), for one target
        or the whole dataset."""
        df = self.get_target(target_id) if target_id else self.ground_truth
        return float(df["Time"].max() - df["Time"].min())

    def sample_rate(self, target_id: str | None = None) -> float:
        """Return the constant timestep between consecutive samples, for one
        target (or the first target, if none is given)."""
        target_id = target_id or self.list_targets()[0]
        df = self.get_target(target_id)
        deltas = df["Time"].diff().dropna()
        return float(deltas.iloc[0]) if not deltas.empty else float("nan")

    def trajectory_length(self, target_id: str) -> float:
        """Return the total 3D path length flown by one target, in the same
        distance units as X/Y/Z (typically meters)."""
        df = self.get_target(target_id)
        deltas = df[["X", "Y", "Z"]].diff().dropna()
        segment_lengths = np.sqrt((deltas ** 2).sum(axis=1))
        return float(segment_lengths.sum())

    def _require_target(self, target_id: str) -> None:
        if target_id not in self.ground_truth["TargetID"].unique():
            raise KeyError(f"Unknown TargetID: {target_id}")
