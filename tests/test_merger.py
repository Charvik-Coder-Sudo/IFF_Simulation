"""Tests for GroundTruthMerger."""

from __future__ import annotations

from pathlib import Path

from iff_simulator.ground_truth import REQUIRED_COLUMNS, GroundTruthLoader, GroundTruthMerger


def test_merge_combines_all_targets(aircrafts_dir: Path) -> None:
    scenario = GroundTruthLoader(aircrafts_dir).load()
    merged = GroundTruthMerger(scenario).merge()
    assert list(merged.columns) == REQUIRED_COLUMNS
    assert len(merged) == 5 + 4
    assert set(merged["TargetID"].unique()) == {"TARGET_1", "TARGET_2"}


def test_merge_sorts_by_time_then_target_id(aircrafts_dir: Path) -> None:
    scenario = GroundTruthLoader(aircrafts_dir).load()
    merged = GroundTruthMerger(scenario).merge()
    sorted_check = merged.sort_values(["Time", "TargetID"]).reset_index(drop=True)
    assert merged.equals(sorted_check)


def test_save_writes_csv(aircrafts_dir: Path, tmp_path: Path) -> None:
    scenario = GroundTruthLoader(aircrafts_dir).load()
    merger = GroundTruthMerger(scenario)
    merged = merger.merge()
    output_path = tmp_path / "nested" / "ground_truth.csv"
    saved_path = merger.save(merged, output_path)
    assert saved_path == output_path
    assert output_path.exists()
