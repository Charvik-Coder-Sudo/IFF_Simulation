"""Tests for GroundTruthStatistics."""

from __future__ import annotations

from pathlib import Path

from iff_simulator.ground_truth import (
    GroundTruthInspector,
    GroundTruthLoader,
    GroundTruthStatistics,
)


def test_compute_returns_one_row_per_target(aircrafts_dir: Path) -> None:
    scenario = GroundTruthLoader(aircrafts_dir).load()
    inspector = GroundTruthInspector(scenario)
    stats_df = GroundTruthStatistics(inspector).compute()

    assert set(stats_df["TargetID"]) == {"TARGET_1", "TARGET_2"}
    expected_columns = {
        "TargetID",
        "MaxSpeed",
        "AvgSpeed",
        "MaxAltitude",
        "MinAltitude",
        "TotalDistance",
        "FlightDuration",
    }
    assert expected_columns.issubset(stats_df.columns)


def test_save_writes_csv(aircrafts_dir: Path, tmp_path: Path) -> None:
    scenario = GroundTruthLoader(aircrafts_dir).load()
    inspector = GroundTruthInspector(scenario)
    stats = GroundTruthStatistics(inspector)
    stats_df = stats.compute()

    output_path = tmp_path / "statistics.csv"
    saved_path = stats.save(stats_df, output_path)
    assert saved_path.exists()
