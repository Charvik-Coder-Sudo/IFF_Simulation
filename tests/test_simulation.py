"""Tests for the simulation package: SimulationClock, WorldState."""

from __future__ import annotations

from iff_simulator.domain import Aircraft, AircraftState, Scenario, Vector3
from iff_simulator.simulation import SimulationClock, WorldState


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
    aircraft = [Aircraft(aircraft_id="TARGET_1")]
    history = {
        "TARGET_1": [
            AircraftState(time=0.0, position=Vector3(0, 0, 0), velocity=Vector3(1, 0, 0)),
            AircraftState(time=1.0, position=Vector3(1, 0, 0), velocity=Vector3(1, 0, 0)),
            AircraftState(time=2.0, position=Vector3(2, 0, 0), velocity=Vector3(1, 0, 0)),
        ]
    }
    return Scenario(aircraft, history)


def test_world_state_update_advances_current_state():
    scenario = _build_scenario()
    clock = SimulationClock(start_time=0.0, dt=1.0, end_time=2.0)
    world = WorldState(scenario, clock)

    assert world.current_state()["TARGET_1"].time == 0.0
    world.update()
    assert world.current_state()["TARGET_1"].time == 1.0
    world.update()
    assert world.current_state()["TARGET_1"].time == 2.0


def test_world_state_no_motion_propagation():
    """update() must only move the current-state cursor, never fabricate
    or interpolate a new AircraftState."""
    scenario = _build_scenario()
    clock = SimulationClock(start_time=0.0, dt=1.0, end_time=2.0)
    world = WorldState(scenario, clock)

    world.update()
    current = world.current_state()["TARGET_1"]
    recorded = scenario.get_state_history("TARGET_1")[1]
    assert current is recorded
