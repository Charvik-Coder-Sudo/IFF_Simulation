"""Tests for SelectionPolicy / DefaultSelectionPolicy."""

from __future__ import annotations

import pytest

from iff_simulator.domain import Aircraft, AircraftState, Scenario, Vector3
from iff_simulator.geometry import RelativeState
from iff_simulator.sensors.iff import DefaultSelectionPolicy, Ownship, SelectionPolicy

ZERO = Vector3(0.0, 0.0, 0.0)


def _make_ownship(maximum_range: float = 1000.0, beam_width: float = 60.0, beam_height: float = 20.0) -> Ownship:
    scenario = Scenario(
        [Aircraft(aircraft_id="OWNSHIP")],
        {"OWNSHIP": [AircraftState(time=0.0, position=ZERO, velocity=ZERO)]},
    )
    return Ownship(
        aircraft_id="OWNSHIP",
        scenario=scenario,
        maximum_range=maximum_range,
        beam_width=beam_width,
        beam_height=beam_height,
    )


def _make_relative_state(range_m: float = 100.0, azimuth_deg: float = 0.0, elevation_deg: float = 0.0) -> RelativeState:
    return RelativeState(
        target_id="T1",
        time=0.0,
        relative_position=Vector3(range_m, 0.0, 0.0),
        relative_velocity=ZERO,
        range_m=range_m,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        bearing_deg=0.0,
        closing_velocity_mps=0.0,
    )


def _make_aircraft(iff_capability: str = "MODE_4") -> Aircraft:
    return Aircraft(aircraft_id="T1", iff_capability=iff_capability)


def _make_aircraft_state(alive: bool = True) -> AircraftState:
    return AircraftState(time=0.0, position=Vector3(100, 0, 0), velocity=ZERO, alive=alive)


def test_selection_policy_is_abstract():
    with pytest.raises(TypeError):
        SelectionPolicy()  # type: ignore[abstract]


def test_default_policy_accepts_alive_capable_in_range_in_beam_target():
    policy = DefaultSelectionPolicy()
    ownship = _make_ownship()
    relative_state = _make_relative_state(range_m=100.0, azimuth_deg=0.0, elevation_deg=0.0)
    assert policy.accept(relative_state, _make_aircraft(), _make_aircraft_state(alive=True), ownship) is True


def test_default_policy_rejects_dead_aircraft():
    policy = DefaultSelectionPolicy()
    ownship = _make_ownship()
    relative_state = _make_relative_state()
    assert policy.accept(relative_state, _make_aircraft(), _make_aircraft_state(alive=False), ownship) is False


def test_default_policy_rejects_iff_incapable_aircraft():
    policy = DefaultSelectionPolicy()
    ownship = _make_ownship()
    relative_state = _make_relative_state()
    aircraft = _make_aircraft(iff_capability="UNKNOWN")
    assert policy.accept(relative_state, aircraft, _make_aircraft_state(), ownship) is False


def test_default_policy_rejects_outside_maximum_range():
    policy = DefaultSelectionPolicy()
    ownship = _make_ownship(maximum_range=1000.0)
    relative_state = _make_relative_state(range_m=1000.1)
    assert policy.accept(relative_state, _make_aircraft(), _make_aircraft_state(), ownship) is False


def test_default_policy_rejects_outside_beam_azimuth():
    policy = DefaultSelectionPolicy()
    ownship = _make_ownship(beam_width=60.0)  # +/- 30 degrees
    relative_state = _make_relative_state(azimuth_deg=30.1)
    assert policy.accept(relative_state, _make_aircraft(), _make_aircraft_state(), ownship) is False


def test_default_policy_rejects_outside_beam_elevation():
    policy = DefaultSelectionPolicy()
    ownship = _make_ownship(beam_height=20.0)  # +/- 10 degrees
    relative_state = _make_relative_state(elevation_deg=10.1)
    assert policy.accept(relative_state, _make_aircraft(), _make_aircraft_state(), ownship) is False


def test_default_policy_boundary_exactly_on_range_edge_is_accepted():
    policy = DefaultSelectionPolicy()
    ownship = _make_ownship(maximum_range=1000.0)
    relative_state = _make_relative_state(range_m=1000.0)
    assert policy.accept(relative_state, _make_aircraft(), _make_aircraft_state(), ownship) is True


def test_default_policy_boundary_exactly_on_beam_azimuth_edge_is_accepted():
    policy = DefaultSelectionPolicy()
    ownship = _make_ownship(beam_width=60.0)  # half-width == 30.0
    relative_state = _make_relative_state(azimuth_deg=30.0)
    assert policy.accept(relative_state, _make_aircraft(), _make_aircraft_state(), ownship) is True


def test_default_policy_boundary_exactly_on_beam_elevation_edge_is_accepted():
    policy = DefaultSelectionPolicy()
    ownship = _make_ownship(beam_height=20.0)  # half-height == 10.0
    relative_state = _make_relative_state(elevation_deg=10.0)
    assert policy.accept(relative_state, _make_aircraft(), _make_aircraft_state(), ownship) is True


def test_default_policy_negative_azimuth_and_elevation_use_absolute_value():
    policy = DefaultSelectionPolicy()
    ownship = _make_ownship(beam_width=60.0, beam_height=20.0)
    relative_state = _make_relative_state(azimuth_deg=-30.0, elevation_deg=-10.0)
    assert policy.accept(relative_state, _make_aircraft(), _make_aircraft_state(), ownship) is True

    relative_state_outside = _make_relative_state(azimuth_deg=-30.1, elevation_deg=0.0)
    assert policy.accept(relative_state_outside, _make_aircraft(), _make_aircraft_state(), ownship) is False
