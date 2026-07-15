"""Merges every aircraft's recorded state history into one ground-truth table.

Purpose:
    Implements `GroundTruthMerger`, which flattens a `Scenario`'s
    per-aircraft `AircraftState` histories into a single, sorted table
    and writes it to `ground_truth.csv`.

Inputs:
    A `Scenario`, as produced by `GroundTruthLoader.load()` and checked
    by `GroundTruthValidator`.

Outputs:
    A single merged `pandas.DataFrame`, sorted by (Time, TargetID), and
    optionally written to a CSV file.

Engineering explanation:
    CSV is a file format, not a runtime data structure, so building a
    DataFrame here — purely as a serialization step for `to_csv` — does
    not reintroduce the DataFrame-passing anti-pattern this refactor
    removes: `Scenario`/`AircraftState` remain the only representation
    passed between runtime modules. Values are taken verbatim from each
    `AircraftState` (no recomputation), which is what guarantees
    `ground_truth.csv` stays byte-for-byte identical to the pre-refactor
    output.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..domain import Scenario
from .models import REQUIRED_COLUMNS


class GroundTruthMerger:
    """Merges a Scenario's per-aircraft state histories into one sorted table.

    Purpose:
        Combine every aircraft's recorded state history into the single
        canonical ground-truth dataset used by every other module in
        this subsystem, and persist it as `ground_truth.csv`.

    Inputs:
        `scenario`: a `Scenario` containing one recorded state history
        per aircraft.

    Outputs:
        `merge()` returns the combined, sorted DataFrame.
        `save()` writes that DataFrame to a CSV file at a given path.

    Engineering explanation:
        Sorting by (Time, TargetID) makes the merged table directly
        usable as a scan-by-scan feed: iterating rows in order naturally
        groups all aircraft present at the same Time together.
    """

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario

    def merge(self) -> pd.DataFrame:
        """Flatten and sort every aircraft's state history into one DataFrame."""
        aircraft_ids = self.scenario.list_aircraft_ids()
        if not aircraft_ids:
            raise ValueError("No trajectories to merge.")

        records = []
        for aircraft_id in aircraft_ids:
            for state in self.scenario.get_state_history(aircraft_id):
                records.append(
                    {
                        "Time": state.time,
                        "TargetID": aircraft_id,
                        "X": state.position.x,
                        "Y": state.position.y,
                        "Z": state.position.z,
                        "VX": state.velocity.x,
                        "VY": state.velocity.y,
                        "VZ": state.velocity.z,
                        "Range": state.range_m,
                        "Azimuth": state.azimuth_deg,
                        "Elevation": state.elevation_deg,
                    }
                )

        merged = pd.DataFrame.from_records(records, columns=REQUIRED_COLUMNS)
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
