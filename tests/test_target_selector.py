"""Tests for TargetSelector."""

from __future__ import annotations

import time as time_module

import pytest

from iff_simulator.domain import Aircraft, AircraftState, Scenario, Vector3
from iff_simulator.geometry import RelativeState
from iff_simulator.sensors.iff import SelectedTarget, TargetSelector
from iff_simulator.simulation import SimulationClock, World

ZERO = Vector3(0.0, 0.0, 0.0)
OWNSHIP_ID = "OWNSHIP"

DEFAULT_OWNSHIP_KWARGS = dict(maximum_range=1000.0, beam_width=60.0, beam_height=20.0)


def _build_world(target_specs: dict[str, tuple[bool, str, Vector3]], **ownship_kwargs) -> World:
    """target_specs: target_id -> (alive, iff_capability, position)."""
    aircraft_list = [Aircraft(aircraft_id=OWNSHIP_ID)]
    history = {OWNSHIP_ID: [AircraftState(time=0.0, position=ZERO, velocity=ZERO)]}
    for target_id, (alive, iff_capability, position) in target_specs.items():
        aircraft_list.append(Aircraft(aircraft_id=target_id, iff_capability=iff_capability))
        history[target_id] = [
            AircraftState(time=0.0, position=position, velocity=ZERO, alive=alive)
        ]
    scenario = Scenario(aircraft_list, history)
    clock = SimulationClock(start_time=0.0, dt=1.0, end_time=0.0)
    merged_kwargs = {**DEFAULT_OWNSHIP_KWARGS, **ownship_kwargs}
    return World(scenario, clock, ownship_id=OWNSHIP_ID, **merged_kwargs)


class _FakeGeometryEngine:
    """Test double for GeometryEngine: returns pre-registered RelativeState
    values keyed by target_id, bypassing real trigonometry entirely so exact
    boundary conditions (azimuth/elevation/range == the limit) can be tested
    deterministically, without floating-point placement error. Demonstrates
    TargetSelector's geometry_engine dependency injection."""

    def __init__(self, states_by_id: dict[str, RelativeState]) -> None:
        self._states_by_id = states_by_id

    def compute_batch(self, time, ownship_position, ownship_velocity, ownship_heading, targets):
        return [self._states_by_id[target_id] for target_id, _, _ in targets]

    def compute_relative_state(
        self, target_id, time, ownship_position, ownship_velocity, ownship_heading, target_position, target_velocity
    ):
        return self._states_by_id[target_id]


def _fake_relative_state(target_id: str, range_m: float, azimuth_deg: float = 0.0, elevation_deg: float = 0.0) -> RelativeState:
    return RelativeState(
        target_id=target_id,
        time=0.0,
        relative_position=Vector3(range_m, 0.0, 0.0),
        relative_velocity=ZERO,
        range_m=range_m,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        bearing_deg=0.0,
        closing_velocity_mps=0.0,
    )


# ---------------------------------------------------------------------------
# Real GeometryEngine, end-to-end
# ---------------------------------------------------------------------------


def test_alive_target_selected():
    world = _build_world({"T1": (True, "MODE_4", Vector3(100, 0, 0))})
    selector = TargetSelector(world)
    assert selector.visible_targets() == ["T1"]


def test_dead_target_rejected():
    world = _build_world({"T1": (False, "MODE_4", Vector3(100, 0, 0))})
    selector = TargetSelector(world)
    assert selector.visible_targets() == []


def test_iff_incapable_target_rejected():
    world = _build_world({"T1": (True, "UNKNOWN", Vector3(100, 0, 0))})
    selector = TargetSelector(world)
    assert selector.visible_targets() == []


def test_outside_range_rejected():
    world = _build_world({"T1": (True, "MODE_4", Vector3(5000, 0, 0))}, maximum_range=1000.0)
    selector = TargetSelector(world)
    assert selector.visible_targets() == []


def test_outside_beam_rejected():
    # 100 m north-east-ish: azimuth ~45 degrees, well outside a 60-degree beam (+/-30).
    world = _build_world({"T1": (True, "MODE_4", Vector3(100, 100, 0))}, beam_width=60.0)
    selector = TargetSelector(world)
    assert selector.visible_targets() == []


def test_multiple_visible_targets():
    world = _build_world(
        {
            "T1": (True, "MODE_4", Vector3(100, 0, 0)),  # selected
            "T2": (False, "MODE_4", Vector3(150, 0, 0)),  # dead -> rejected
            "T3": (True, "UNKNOWN", Vector3(200, 0, 0)),  # not iff-capable -> rejected
            "T4": (True, "MODE_4", Vector3(5000, 0, 0)),  # out of range -> rejected
            "T5": (True, "MODE_4", Vector3(50, 50, 0)),  # out of beam -> rejected
            "T6": (True, "MODE_5", Vector3(300, 0, 0)),  # selected
        }
    )
    selector = TargetSelector(world)
    assert selector.visible_targets() == ["T1", "T6"]


def test_empty_world_returns_empty_list():
    world = _build_world({})
    selector = TargetSelector(world)
    assert selector.select_targets() == []
    assert selector.visible_targets() == []


def test_ownship_never_selected():
    world = _build_world({"T1": (True, "MODE_4", Vector3(100, 0, 0))})
    selector = TargetSelector(world)
    assert OWNSHIP_ID not in selector.visible_targets()
    with pytest.raises(KeyError):
        selector.select_one(OWNSHIP_ID)


def test_select_one_returns_selected_target_when_it_passes():
    world = _build_world({"T1": (True, "MODE_4", Vector3(100, 0, 0))})
    selector = TargetSelector(world)
    result = selector.select_one("T1")
    assert isinstance(result, SelectedTarget)
    assert result.target_id == "T1"
    assert result.range_m == pytest.approx(100.0)


def test_select_one_returns_none_when_it_fails_policy():
    world = _build_world({"T1": (False, "MODE_4", Vector3(100, 0, 0))})
    selector = TargetSelector(world)
    assert selector.select_one("T1") is None


def test_select_one_unknown_target_raises_key_error():
    world = _build_world({"T1": (True, "MODE_4", Vector3(100, 0, 0))})
    selector = TargetSelector(world)
    with pytest.raises(KeyError):
        selector.select_one("NO_SUCH_TARGET")


def test_deterministic_ordering_by_range_then_id():
    world = _build_world(
        {
            "T3": (True, "MODE_4", Vector3(300, 0, 0)),
            "T1": (True, "MODE_4", Vector3(100, 0, 0)),
            "T2": (True, "MODE_4", Vector3(200, 0, 0)),
        }
    )
    selector = TargetSelector(world)
    assert selector.visible_targets() == ["T1", "T2", "T3"]


def test_deterministic_ordering_ties_on_range_break_by_target_id():
    world = _build_world(
        {
            "B": (True, "MODE_4", Vector3(100, 0, 0)),
            "A": (True, "MODE_4", Vector3(100, 0, 0)),
        }
    )
    selector = TargetSelector(world)
    assert selector.visible_targets() == ["A", "B"]


def test_target_selector_never_computes_geometry_itself():
    """TargetSelector must call into GeometryEngine, not recompute range/azimuth
    itself: swapping in a fake engine that returns fixed values must fully
    determine the output, proving no independent geometry math exists in
    TargetSelector."""
    world = _build_world({"T1": (True, "MODE_4", Vector3(999_999, 0, 0))})  # would fail real geometry checks
    fake_engine = _FakeGeometryEngine({"T1": _fake_relative_state("T1", range_m=50.0)})
    selector = TargetSelector(world, geometry_engine=fake_engine)
    result = selector.select_one("T1")
    assert result is not None
    assert result.range_m == 50.0  # came entirely from the fake engine


# ---------------------------------------------------------------------------
# Exact boundary conditions (via fake GeometryEngine, for precision)
# ---------------------------------------------------------------------------


def test_boundary_exactly_on_range_edge_is_selected():
    world = _build_world({"T1": (True, "MODE_4", Vector3(1000, 0, 0))}, maximum_range=1000.0)
    fake_engine = _FakeGeometryEngine({"T1": _fake_relative_state("T1", range_m=1000.0)})
    selector = TargetSelector(world, geometry_engine=fake_engine)
    assert selector.visible_targets() == ["T1"]


def test_boundary_exactly_on_beam_edge_is_selected():
    world = _build_world({"T1": (True, "MODE_4", Vector3(100, 0, 0))}, beam_width=60.0, beam_height=20.0)
    fake_engine = _FakeGeometryEngine(
        {"T1": _fake_relative_state("T1", range_m=100.0, azimuth_deg=30.0, elevation_deg=10.0)}
    )
    selector = TargetSelector(world, geometry_engine=fake_engine)
    assert selector.visible_targets() == ["T1"]


def test_just_outside_range_edge_is_rejected():
    world = _build_world({"T1": (True, "MODE_4", Vector3(1000, 0, 0))}, maximum_range=1000.0)
    fake_engine = _FakeGeometryEngine({"T1": _fake_relative_state("T1", range_m=1000.0000001)})
    selector = TargetSelector(world, geometry_engine=fake_engine)
    assert selector.visible_targets() == []


def test_just_outside_beam_edge_is_rejected():
    world = _build_world({"T1": (True, "MODE_4", Vector3(100, 0, 0))}, beam_width=60.0)
    fake_engine = _FakeGeometryEngine({"T1": _fake_relative_state("T1", range_m=100.0, azimuth_deg=30.0000001)})
    selector = TargetSelector(world, geometry_engine=fake_engine)
    assert selector.visible_targets() == []


# ---------------------------------------------------------------------------
# Performance: O(N), no nested search
# ---------------------------------------------------------------------------


def test_batch_selection_performance_scales_linearly_enough():
    target_count = 3000
    specs = {
        f"T{i}": (True, "MODE_4", Vector3(100 + i, 0, 0))
        for i in range(target_count)
    }
    world = _build_world(specs, maximum_range=10_000.0)
    selector = TargetSelector(world)

    start = time_module.perf_counter()
    results = selector.select_targets()
    elapsed = time_module.perf_counter() - start

    assert len(results) == target_count
    assert elapsed < 2.0  # generous bound; a quadratic implementation would be far slower
