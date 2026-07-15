"""Tests for GroundTruthLoader."""

from __future__ import annotations

from pathlib import Path

import pytest

from iff_simulator.domain import Scenario
from iff_simulator.ground_truth import GroundTruthLoader


def test_discover_files_finds_all_tdf(aircrafts_dir: Path) -> None:
    loader = GroundTruthLoader(aircrafts_dir)
    files = loader.discover_files()
    assert len(files) == 2
    assert all(f.suffix == ".tdf" for f in files)


def test_load_returns_scenario_with_expected_aircraft(aircrafts_dir: Path) -> None:
    loader = GroundTruthLoader(aircrafts_dir)
    scenario = loader.load()
    assert isinstance(scenario, Scenario)
    assert scenario.list_aircraft_ids() == ["TARGET_1", "TARGET_2"]
    assert len(scenario.get_state_history("TARGET_1")) == 5
    assert len(scenario.get_state_history("TARGET_2")) == 4


def test_load_populates_aircraft_metadata(aircrafts_dir: Path) -> None:
    loader = GroundTruthLoader(aircrafts_dir)
    scenario = loader.load()
    aircraft = scenario.get_aircraft("TARGET_1")
    assert aircraft.aircraft_id == "TARGET_1"


def test_load_builds_correct_state_values(aircrafts_dir: Path) -> None:
    loader = GroundTruthLoader(aircrafts_dir)
    scenario = loader.load()
    first_state = scenario.get_state_history("TARGET_1")[0]
    assert first_state.time == 1
    assert first_state.position.x == 0.0
    assert first_state.position.y == 0.0
    assert first_state.velocity.x == 10.0


def test_missing_directory_raises() -> None:
    with pytest.raises(FileNotFoundError):
        GroundTruthLoader("does/not/exist")


def test_empty_directory_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    loader = GroundTruthLoader(empty)
    with pytest.raises(FileNotFoundError):
        loader.discover_files()
