"""Tests for Ownship."""

from __future__ import annotations

from iff_simulator.domain import Aircraft, AircraftState, Scenario, Vector3
from iff_simulator.sensors.iff import Ownship


def _build_scenario() -> Scenario:
    aircraft = [Aircraft(aircraft_id="TARGET_1")]
    history = {
        "TARGET_1": [
            AircraftState(
                time=0.0,
                position=Vector3(0, 0, 0),
                velocity=Vector3(1, 0, 0),
                heading=90.0,
            ),
            AircraftState(
                time=1.0,
                position=Vector3(1, 0, 0),
                velocity=Vector3(2, 0, 0),
                heading=95.0,
            ),
        ]
    }
    return Scenario(aircraft, history)


def test_ownship_defaults():
    scenario = _build_scenario()
    ownship = Ownship(aircraft_id="TARGET_1", scenario=scenario)
    assert ownship.pitch == 0.0
    assert ownship.roll == 0.0
    assert ownship.maximum_range == 0.0
    assert ownship.beam_width == 0.0
    assert ownship.beam_height == 0.0
    assert ownship.interrogation_rate == 0.0
    assert ownship.operating_modes == []


def test_ownship_configuration_fields_stored():
    scenario = _build_scenario()
    ownship = Ownship(
        aircraft_id="TARGET_1",
        scenario=scenario,
        pitch=1.5,
        roll=-2.0,
        maximum_range=50000.0,
        beam_width=4.0,
        beam_height=3.0,
        interrogation_rate=20.0,
        operating_modes=["MODE_3A", "MODE_C"],
    )
    assert ownship.pitch == 1.5
    assert ownship.roll == -2.0
    assert ownship.maximum_range == 50000.0
    assert ownship.beam_width == 4.0
    assert ownship.beam_height == 3.0
    assert ownship.interrogation_rate == 20.0
    assert ownship.operating_modes == ["MODE_3A", "MODE_C"]


def test_ownship_position_velocity_heading_read_through_scenario():
    scenario = _build_scenario()
    ownship = Ownship(aircraft_id="TARGET_1", scenario=scenario)

    assert ownship.position == Vector3(0, 0, 0)
    assert ownship.velocity == Vector3(1, 0, 0)
    assert ownship.heading == 90.0


def test_ownship_reflects_scenario_cursor_advance_without_manual_sync():
    scenario = _build_scenario()
    ownship = Ownship(aircraft_id="TARGET_1", scenario=scenario)

    scenario.set_current_index("TARGET_1", 1)

    assert ownship.position == Vector3(1, 0, 0)
    assert ownship.velocity == Vector3(2, 0, 0)
    assert ownship.heading == 95.0


def test_ownship_never_duplicates_state():
    scenario = _build_scenario()
    ownship = Ownship(aircraft_id="TARGET_1", scenario=scenario)

    assert ownship.position is scenario.get_state("TARGET_1").position
