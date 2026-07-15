"""Tests for GroundTruthInspector."""

from __future__ import annotations

from pathlib import Path

import pytest

from iff_simulator.ground_truth import GroundTruthInspector, GroundTruthLoader


@pytest.fixture
def inspector(aircrafts_dir: Path) -> GroundTruthInspector:
    scenario = GroundTruthLoader(aircrafts_dir).load()
    return GroundTruthInspector(scenario)


def test_list_targets(inspector: GroundTruthInspector) -> None:
    assert inspector.list_targets() == ["TARGET_1", "TARGET_2"]


def test_get_target_returns_only_that_targets_history(inspector: GroundTruthInspector) -> None:
    history = inspector.get_target("TARGET_1")
    assert len(history) == 5
    assert all(state.time is not None for state in history)


def test_get_target_unknown_raises(inspector: GroundTruthInspector) -> None:
    with pytest.raises(KeyError):
        inspector.get_target("TARGET_99")


def test_get_time_returns_all_targets_at_that_time(inspector: GroundTruthInspector) -> None:
    states_at_time = inspector.get_time(1)
    assert set(states_at_time.keys()) == {"TARGET_1", "TARGET_2"}


def test_summary_contains_expected_keys(inspector: GroundTruthInspector) -> None:
    summary = inspector.summary()
    assert summary["target_count"] == 2
    assert summary["total_rows"] == 9


def test_bounding_box_whole_dataset(inspector: GroundTruthInspector) -> None:
    box = inspector.bounding_box()
    assert box["x_min"] <= box["x_max"]
    assert box["z_min"] == 100.0
    assert box["z_max"] == 100.0


def test_duration_per_target(inspector: GroundTruthInspector) -> None:
    assert inspector.duration("TARGET_1") == 4.0
    assert inspector.duration("TARGET_2") == 3.0


def test_sample_rate_is_constant_timestep(inspector: GroundTruthInspector) -> None:
    assert inspector.sample_rate("TARGET_1") == 1.0


def test_trajectory_length_is_nonnegative(inspector: GroundTruthInspector) -> None:
    assert inspector.trajectory_length("TARGET_1") > 0
