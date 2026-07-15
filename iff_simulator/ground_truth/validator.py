"""Validates loaded ground-truth trajectories before they are trusted downstream.

Purpose:
    Implements `GroundTruthValidator`, which runs a fixed set of
    integrity checks against every trajectory loaded by
    `GroundTruthLoader`, before any merging, inspection, or visualization
    happens.

Inputs:
    `dict[str, pandas.DataFrame]` as produced by `GroundTruthLoader.load()`.

Outputs:
    A printed validation report. Raises `GroundTruthValidationError` with
    a descriptive message the moment a hard-integrity check fails.

Engineering explanation:
    Checks are split into "hard" checks (missing columns, missing
    values, wrong dtypes, non-increasing timestamps, non-constant
    per-target timestep) which raise immediately, and "informational"
    checks (sample count differences across targets) which are only
    reported, because different aircraft legitimately fly for different
    durations and therefore produce different sample counts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models import REQUIRED_COLUMNS

_NUMERIC_COLUMNS = [c for c in REQUIRED_COLUMNS if c != "TargetID"]


class GroundTruthValidationError(Exception):
    """Raised when a loaded ground-truth trajectory fails a hard integrity check."""


class GroundTruthValidator:
    """Runs integrity checks over a set of loaded ground-truth trajectories.

    Purpose:
        Guarantee that every trajectory handed to the merger, inspector,
        and visualization modules is well-formed, so those modules never
        need to defend against malformed input themselves.

    Inputs:
        `trajectories`: dict mapping TargetID -> raw trajectory DataFrame,
        as produced by `GroundTruthLoader.load()`.

    Outputs:
        `validate()` prints a human-readable report and returns `None`
        on success, or raises `GroundTruthValidationError` on the first
        hard-integrity failure.

    Engineering explanation:
        Validation is deliberately fail-fast: a simulator feeding bad
        ground truth into a sensor-fusion pipeline is worse than one
        that simply refuses to run, since downstream tracking/IFF logic
        would silently produce misleading results otherwise.
    """

    def __init__(self, trajectories: dict[str, pd.DataFrame]) -> None:
        self.trajectories = trajectories

    def validate(self, verbose: bool = True) -> None:
        """Run all validation checks, printing a report as it goes.

        Raises:
            GroundTruthValidationError: on the first hard-integrity failure.
        """
        report_lines = ["Ground Truth Validation Report", "=" * 40]

        for target_id, df in self.trajectories.items():
            report_lines.append(f"\nTarget: {target_id}")
            self._check_required_columns(target_id, df)
            self._check_missing_values(target_id, df)
            self._check_dtypes(target_id, df)
            self._check_increasing_timestamps(target_id, df)
            timestep = self._check_constant_timestep(target_id, df)
            report_lines.append(f"  Rows: {len(df)}")
            report_lines.append(f"  Timestep: {timestep}")
            report_lines.append("  Status: OK")

        self._report_sample_counts(report_lines)

        report_lines.append("\nAll hard integrity checks passed.")
        if verbose:
            print("\n".join(report_lines))

    def _check_required_columns(self, target_id: str, df: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise GroundTruthValidationError(
                f"{target_id}: missing required columns {missing}"
            )

    def _check_missing_values(self, target_id: str, df: pd.DataFrame) -> None:
        if df[REQUIRED_COLUMNS].isnull().values.any():
            null_columns = df.columns[df.isnull().any()].tolist()
            raise GroundTruthValidationError(
                f"{target_id}: missing values found in columns {null_columns}"
            )

    def _check_dtypes(self, target_id: str, df: pd.DataFrame) -> None:
        for column in _NUMERIC_COLUMNS:
            if not np.issubdtype(df[column].dtype, np.number):
                raise GroundTruthValidationError(
                    f"{target_id}: column '{column}' expected a numeric dtype, "
                    f"got {df[column].dtype}"
                )

    def _check_increasing_timestamps(self, target_id: str, df: pd.DataFrame) -> None:
        if not df["Time"].is_monotonic_increasing:
            raise GroundTruthValidationError(
                f"{target_id}: Time column is not strictly increasing"
            )

    def _check_constant_timestep(self, target_id: str, df: pd.DataFrame) -> float:
        deltas = df["Time"].diff().dropna().round(6)
        unique_deltas = deltas.unique()
        if len(unique_deltas) > 1:
            raise GroundTruthValidationError(
                f"{target_id}: timestep is not constant, found values "
                f"{sorted(unique_deltas)[:5]}"
            )
        return float(unique_deltas[0]) if len(unique_deltas) else float("nan")

    def _report_sample_counts(self, report_lines: list[str]) -> None:
        counts = {tid: len(df) for tid, df in self.trajectories.items()}
        report_lines.append(f"\nSample counts per target: {counts}")
        if len(set(counts.values())) > 1:
            report_lines.append(
                "  Note: sample counts differ across targets - expected, "
                "since aircraft fly for different durations."
            )
