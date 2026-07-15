"""Tests for GroundTruthValidator."""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path

import pytest

from iff_simulator.domain import Scenario
from iff_simulator.ground_truth import GroundTruthLoader
from iff_simulator.ground_truth.validator import (
    GroundTruthValidationError,
    GroundTruthValidator,
)


def _scenario_with_target_1_replaced(aircrafts_dir: Path, mutate) -> Scenario:
    """Load the fixture scenario, then return a new Scenario whose
    TARGET_1 state history has been mutated by `mutate(history)`."""
    scenario = GroundTruthLoader(aircrafts_dir).load()
    history = deepcopy(scenario.get_state_history("TARGET_1"))
    mutate(history)
    aircraft_list = scenario.get_all_aircraft()
    state_history = {
        aircraft_id: (history if aircraft_id == "TARGET_1" else scenario.get_state_history(aircraft_id))
        for aircraft_id in scenario.list_aircraft_ids()
    }
    return Scenario(aircraft_list, state_history)


def test_validate_passes_on_well_formed_data(aircrafts_dir: Path) -> None:
    scenario = GroundTruthLoader(aircrafts_dir).load()
    validator = GroundTruthValidator(scenario)
    validator.validate(verbose=False)  # should not raise


def test_validate_raises_on_missing_values(aircrafts_dir: Path) -> None:
    def mutate(history):
        history[0].position = type(history[0].position)(math.nan, history[0].position.y, history[0].position.z)

    scenario = _scenario_with_target_1_replaced(aircrafts_dir, mutate)
    validator = GroundTruthValidator(scenario)
    with pytest.raises(GroundTruthValidationError):
        validator.validate(verbose=False)


def test_validate_raises_on_non_increasing_time(aircrafts_dir: Path) -> None:
    def mutate(history):
        history[1].time = history[0].time

    scenario = _scenario_with_target_1_replaced(aircrafts_dir, mutate)
    validator = GroundTruthValidator(scenario)
    with pytest.raises(GroundTruthValidationError):
        validator.validate(verbose=False)


def test_validate_raises_on_non_constant_timestep(aircrafts_dir: Path) -> None:
    def mutate(history):
        history[2].time = history[2].time + 5

    scenario = _scenario_with_target_1_replaced(aircrafts_dir, mutate)
    validator = GroundTruthValidator(scenario)
    with pytest.raises(GroundTruthValidationError):
        validator.validate(verbose=False)


def test_validate_does_not_raise_on_differing_sample_counts(aircrafts_dir: Path) -> None:
    scenario = GroundTruthLoader(aircrafts_dir).load()
    validator = GroundTruthValidator(scenario)
    validator.validate(verbose=False)  # TARGET_1 has 5 rows, TARGET_2 has 4
