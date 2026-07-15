"""Tests for the simulation package: SimulationClock, World."""

from __future__ import annotations

import pytest

from iff_simulator.domain import Aircraft, AircraftState, Scenario, Vector3
from iff_simulator.simulation import SimulationClock, World


def test_clock_step_advances_by_dt():
    clock = SimulationClock(start_time=0.0, dt=1.0, end_time=10.0)
    assert clock.current_time() == 0.0
    clock.step()
    assert clock.current_time() == 1.0


def test_clock_step_clamps_to_end_time():
    clock = SimulationClock(start_time=0.0, dt=100.0, end_time=10.0)
    clock.step()
    assert clock.current_time() == 10.0


def test_clock_reset_returns_to_start_time():
    clock = SimulationClock(start_time=5.0, dt=1.0, end_time=10.0)
    clock.step()
    clock.step()
    clock.reset()
    assert clock.current_time() == 5.0


def test_clock_finished():
    clock = SimulationClock(start_time=0.0, dt=5.0, end_time=10.0)
    assert not clock.finished()
    clock.step()
    assert not clock.finished()
    clock.step()
    assert clock.finished()


def _build_scenario() -> Scenario:
    aircraft = [
        Aircraft(aircraft_id="TARGET_1"),
        Aircraft(aircraft_id="TARGET_2"),
    ]
    history = {
        "TARGET_1": [
            AircraftState(time=0.0, position=Vector3(0, 0, 0), velocity=Vector3(1, 0, 0)),
            AircraftState(time=1.0, position=Vector3(1, 0, 0), velocity=Vector3(1, 0, 0)),
            AircraftState(time=2.0, position=Vector3(2, 0, 0), velocity=Vector3(1, 0, 0)),
        ],
        "TARGET_2": [
            AircraftState(time=0.0, position=Vector3(5, 5, 0), velocity=Vector3(0, 1, 0)),
            AircraftState(time=1.0, position=Vector3(5, 6, 0), velocity=Vector3(0, 1, 0)),
            AircraftState(time=2.0, position=Vector3(5, 7, 0), velocity=Vector3(0, 1, 0), alive=False),
        ],
    }
    return Scenario(aircraft, history)


def test_world_step_advances_current_time():
    scenario = _build_scenario()
    clock = SimulationClock(start_time=0.0, dt=1.0, end_time=2.0)
    world = World(scenario, clock)

    assert world.current_time() == 0.0
    world.step()
    assert world.current_time() == 1.0


def test_world_step_synchronizes_ownship_from_scenario():
    scenario = _build_scenario()
    clock = SimulationClock(start_time=0.0, dt=1.0, end_time=2.0)
    world = World(scenario, clock)

    assert world.ownship.aircraft_id == "TARGET_1"
    assert world.ownship.position == Vector3(0, 0, 0)
    world.step()
    assert world.ownship.position == Vector3(1, 0, 0)
    world.step()
    assert world.ownship.position == Vector3(2, 0, 0)


def test_world_ownship_never_duplicates_state():
    """Ownship must reference the Scenario's AircraftState, not a copy."""
    scenario = _build_scenario()
    clock = SimulationClock(start_time=0.0, dt=1.0, end_time=2.0)
    world = World(scenario, clock)

    world.step()
    assert world.ownship.position is scenario.get_state("TARGET_1").position


def test_world_no_motion_propagation():
    """step() must only move the current-state cursor, never fabricate
    or interpolate a new AircraftState."""
    scenario = _build_scenario()
    clock = SimulationClock(start_time=0.0, dt=1.0, end_time=2.0)
    world = World(scenario, clock)

    world.step()
    current = scenario.get_state("TARGET_1")
    recorded = scenario.get_state_history("TARGET_1")[1]
    assert current is recorded


def test_world_get_targets_excludes_ownship():
    scenario = _build_scenario()
    clock = SimulationClock(start_time=0.0, dt=1.0, end_time=2.0)
    world = World(scenario, clock)

    target_ids = [aircraft.aircraft_id for aircraft in world.get_targets()]
    assert target_ids == ["TARGET_2"]


def test_world_get_target_rejects_ownship_id():
    scenario = _build_scenario()
    clock = SimulationClock(start_time=0.0, dt=1.0, end_time=2.0)
    world = World(scenario, clock)

    assert world.get_target("TARGET_2").aircraft_id == "TARGET_2"
    with pytest.raises(KeyError):
        world.get_target("TARGET_1")


def test_world_alive_targets_excludes_dead_and_ownship():
    scenario = _build_scenario()
    clock = SimulationClock(start_time=0.0, dt=1.0, end_time=2.0)
    world = World(scenario, clock)

    assert [a.aircraft_id for a in world.alive_targets()] == ["TARGET_2"]
    world.step()
    world.step()  # TARGET_2 becomes not-alive at time=2.0
    assert world.alive_targets() == []


def test_world_iff_capable_targets_empty_when_no_iff_logic():
    scenario = _build_scenario()
    clock = SimulationClock(start_time=0.0, dt=1.0, end_time=2.0)
    world = World(scenario, clock)

    assert world.iff_capable_targets() == []
