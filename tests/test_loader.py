"""Tests for GroundTruthLoader."""

from __future__ import annotations

from pathlib import Path

import pytest

from iff_simulator.ground_truth import REQUIRED_COLUMNS, GroundTruthLoader


def test_discover_files_finds_all_tdf(aircrafts_dir: Path) -> None:
    loader = GroundTruthLoader(aircrafts_dir)
    files = loader.discover_files()
    assert len(files) == 2
    assert all(f.suffix == ".tdf" for f in files)


def test_load_returns_dict_keyed_by_target_id(aircrafts_dir: Path) -> None:
    loader = GroundTruthLoader(aircrafts_dir)
    trajectories = loader.load()
    assert set(trajectories.keys()) == {"TARGET_1", "TARGET_2"}
    assert list(trajectories["TARGET_1"].columns) == REQUIRED_COLUMNS
    assert len(trajectories["TARGET_1"]) == 5
    assert len(trajectories["TARGET_2"]) == 4


def test_load_drops_bogus_tgtid_column_and_fills_target_id(aircrafts_dir: Path) -> None:
    loader = GroundTruthLoader(aircrafts_dir)
    trajectories = loader.load()
    df = trajectories["TARGET_1"]
    assert (df["TargetID"] == "TARGET_1").all()


def test_missing_directory_raises() -> None:
    with pytest.raises(FileNotFoundError):
        GroundTruthLoader("does/not/exist")


def test_empty_directory_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    loader = GroundTruthLoader(empty)
    with pytest.raises(FileNotFoundError):
        loader.discover_files()
