"""Tests for the domain package: Vector3, Aircraft, AircraftState, Scenario."""

from __future__ import annotations

import pytest

from iff_simulator.domain import Aircraft, AircraftState, Scenario, Vector3


def test_vector3_add_sub():
    a = Vector3(1, 2, 3)
    b = Vector3(4, 5, 6)
    assert a + b == Vector3(5, 7, 9)
    assert b - a == Vector3(3, 3, 3)


def test_vector3_scalar_mul_div():
    a = Vector3(2, 4, 6)
    assert a * 2 == Vector3(4, 8, 12)
    assert 2 * a == Vector3(4, 8, 12)
    assert a / 2 == Vector3(1, 2, 3)


def test_vector3_magnitude():
    assert Vector3(3, 4, 0).magnitude() == 5.0


def test_vector3_normalize():
    unit = Vector3(0, 5, 0).normalize()
    assert unit == Vector3(0, 1, 0)


def test_vector3_normalize_zero_raises():
    with pytest.raises(ValueError):
        Vector3(0, 0, 0).normalize()


def test_vector3_dot():
    assert Vector3(1, 0, 0).dot(Vector3(0, 1, 0)) == 0.0
    assert Vector3(1, 2, 3).dot(Vector3(1, 2, 3)) == 14.0


def test_vector3_cross():
    assert Vector3(1, 0, 0).cross(Vector3(0, 1, 0)) == Vector3(0, 0, 1)


def test_vector3_distance_to():
    assert Vector3(0, 0, 0).distance_to(Vector3(3, 4, 0)) == 5.0


def test_vector3_heading_cardinal_directions():
    assert Vector3(0, 1, 0).heading() == pytest.approx(0.0)
    assert Vector3(1, 0, 0).heading() == pytest.approx(90.0)
    assert Vector3(0, -1, 0).heading() == pytest.approx(180.0)
    assert Vector3(-1, 0, 0).heading() == pytest.approx(270.0)


def test_vector3_is_immutable():
    v = Vector3(1, 2, 3)
    with pytest.raises(Exception):
        v.x = 99  # frozen dataclass


def test_aircraft_defaults():
    aircraft = Aircraft(aircraft_id="TARGET_1")
    assert aircraft.aircraft_id == "TARGET_1"
    assert aircraft.identity == "UNKNOWN"
    assert aircraft.iff_capability == "UNKNOWN"
    assert aircraft.mode_data == {}
    assert aircraft.motion_model == "RECORDED_TRAJECTORY"


def test_aircraft_state_defaults():
    state = AircraftState(time=1.0, position=Vector3(0, 0, 0), velocity=Vector3(1, 0, 0))
    assert state.acceleration == Vector3(0.0, 0.0, 0.0)
    assert state.alive is True


def test_scenario_accessors():
    aircraft = [Aircraft(aircraft_id="TARGET_1"), Aircraft(aircraft_id="TARGET_2")]
    history = {
        "TARGET_1": [AircraftState(time=1.0, position=Vector3(0, 0, 0), velocity=Vector3(1, 0, 0))],
        "TARGET_2": [AircraftState(time=1.0, position=Vector3(1, 1, 1), velocity=Vector3(0, 1, 0))],
    }
    scenario = Scenario(aircraft, history)

    assert scenario.list_aircraft_ids() == ["TARGET_1", "TARGET_2"]
    assert scenario.get_aircraft("TARGET_1").aircraft_id == "TARGET_1"
    assert len(scenario.get_all_aircraft()) == 2
    assert scenario.get_state_history("TARGET_2")[0].position == Vector3(1, 1, 1)
    assert scenario.get_state("TARGET_1").time == 1.0


def test_scenario_unknown_aircraft_raises():
    scenario = Scenario([], {})
    with pytest.raises(KeyError):
        scenario.get_aircraft("UNKNOWN")


def test_scenario_set_current_index_updates_get_state():
    aircraft = [Aircraft(aircraft_id="TARGET_1")]
    history = {
        "TARGET_1": [
            AircraftState(time=1.0, position=Vector3(0, 0, 0), velocity=Vector3(1, 0, 0)),
            AircraftState(time=2.0, position=Vector3(1, 0, 0), velocity=Vector3(1, 0, 0)),
        ]
    }
    scenario = Scenario(aircraft, history)
    assert scenario.get_state("TARGET_1").time == 1.0
    scenario.set_current_index("TARGET_1", 1)
    assert scenario.get_state("TARGET_1").time == 2.0


def test_scenario_set_current_index_out_of_range_raises():
    aircraft = [Aircraft(aircraft_id="TARGET_1")]
    history = {"TARGET_1": [AircraftState(time=1.0, position=Vector3(0, 0, 0), velocity=Vector3(1, 0, 0))]}
    scenario = Scenario(aircraft, history)
    with pytest.raises(IndexError):
        scenario.set_current_index("TARGET_1", 5)
