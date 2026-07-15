"""Exhaustive tests for GeometryEngine."""

from __future__ import annotations

import math

import pytest

from iff_simulator.domain import Vector3
from iff_simulator.geometry import GeometryEngine, RelativeState

ZERO = Vector3(0.0, 0.0, 0.0)


def _reference_relative_state(
    ownship_position: Vector3,
    ownship_velocity: Vector3,
    ownship_heading_deg: float,
    target_position: Vector3,
    target_velocity: Vector3,
) -> dict:
    """Independent re-derivation of the spec's formulas, used as an oracle
    for cases (negative/large coordinates) too tedious to hand-compute."""
    dx = target_position.x - ownship_position.x
    dy = target_position.y - ownship_position.y
    dz = target_position.z - ownship_position.z
    r = math.sqrt(dx**2 + dy**2 + dz**2)

    if r == 0.0:
        return {
            "range_m": 0.0,
            "azimuth_deg": 0.0,
            "elevation_deg": 0.0,
            "bearing_deg": 0.0,
            "closing_velocity_mps": 0.0,
        }

    azimuth_deg = math.degrees(math.atan2(dy, dx))
    elevation_deg = math.degrees(math.asin(max(-1.0, min(1.0, dz / r))))
    true_bearing_deg = math.degrees(math.atan2(dx, dy)) % 360.0
    bearing_deg = (true_bearing_deg - ownship_heading_deg) % 360.0

    rvx = target_velocity.x - ownship_velocity.x
    rvy = target_velocity.y - ownship_velocity.y
    rvz = target_velocity.z - ownship_velocity.z
    los = (dx / r, dy / r, dz / r)
    closing_velocity_mps = -(rvx * los[0] + rvy * los[1] + rvz * los[2])

    return {
        "range_m": r,
        "azimuth_deg": azimuth_deg,
        "elevation_deg": elevation_deg,
        "bearing_deg": bearing_deg,
        "closing_velocity_mps": closing_velocity_mps,
    }


def _assert_matches_reference(state: RelativeState, reference: dict) -> None:
    assert state.range_m == pytest.approx(reference["range_m"], abs=1e-9)
    assert state.azimuth_deg == pytest.approx(reference["azimuth_deg"], abs=1e-9)
    assert state.elevation_deg == pytest.approx(reference["elevation_deg"], abs=1e-9)
    assert state.bearing_deg == pytest.approx(reference["bearing_deg"], abs=1e-9)
    assert state.closing_velocity_mps == pytest.approx(reference["closing_velocity_mps"], abs=1e-9)


# ---------------------------------------------------------------------------
# Same position
# ---------------------------------------------------------------------------


def test_same_position_zero_range_and_safe_defaults():
    state = GeometryEngine.compute_relative_state(
        "T1", 0.0, Vector3(0, 0, 0), ZERO, 0.0, Vector3(0, 0, 0), ZERO
    )
    assert state.range_m == 0.0
    assert state.azimuth_deg == 0.0
    assert state.elevation_deg == 0.0
    assert state.bearing_deg == 0.0
    assert state.closing_velocity_mps == 0.0
    assert state.relative_position == Vector3(0, 0, 0)
    assert state.relative_velocity == Vector3(0, 0, 0)


def test_same_position_nonzero_velocity_still_zero_range():
    """Coincident positions with differing velocities must not raise or NaN."""
    state = GeometryEngine.compute_relative_state(
        "T1", 0.0, Vector3(10, 10, 10), Vector3(5, 0, 0), 0.0, Vector3(10, 10, 10), Vector3(-5, 0, 0)
    )
    assert state.range_m == 0.0
    assert state.closing_velocity_mps == 0.0
    assert not math.isnan(state.closing_velocity_mps)


# ---------------------------------------------------------------------------
# Horizontal target (same altitude, due East)
# ---------------------------------------------------------------------------


def test_horizontal_target_due_east():
    state = GeometryEngine.compute_relative_state(
        "T1", 0.0, Vector3(0, 0, 0), ZERO, 0.0, Vector3(100, 0, 0), ZERO
    )
    assert state.range_m == pytest.approx(100.0)
    assert state.azimuth_deg == pytest.approx(0.0)
    assert state.elevation_deg == pytest.approx(0.0)
    assert state.bearing_deg == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# Vertical target (directly above)
# ---------------------------------------------------------------------------


def test_vertical_target_directly_above():
    state = GeometryEngine.compute_relative_state(
        "T1", 0.0, Vector3(0, 0, 0), ZERO, 0.0, Vector3(0, 0, 100), ZERO
    )
    assert state.range_m == pytest.approx(100.0)
    assert state.elevation_deg == pytest.approx(90.0)


def test_vertical_target_directly_below():
    state = GeometryEngine.compute_relative_state(
        "T1", 0.0, Vector3(0, 0, 0), ZERO, 0.0, Vector3(0, 0, -100), ZERO
    )
    assert state.range_m == pytest.approx(100.0)
    assert state.elevation_deg == pytest.approx(-90.0)


# ---------------------------------------------------------------------------
# 45-degree target
# ---------------------------------------------------------------------------


def test_45_degree_target_horizontal_plane():
    state = GeometryEngine.compute_relative_state(
        "T1", 0.0, Vector3(0, 0, 0), ZERO, 0.0, Vector3(100, 100, 0), ZERO
    )
    assert state.range_m == pytest.approx(100.0 * math.sqrt(2))
    assert state.azimuth_deg == pytest.approx(45.0)
    assert state.elevation_deg == pytest.approx(0.0)
    assert state.bearing_deg == pytest.approx(45.0)


def test_45_degree_target_elevation():
    horizontal_distance = 100.0
    state = GeometryEngine.compute_relative_state(
        "T1", 0.0, Vector3(0, 0, 0), ZERO, 0.0, Vector3(horizontal_distance, 0, horizontal_distance), ZERO
    )
    assert state.elevation_deg == pytest.approx(45.0)


# ---------------------------------------------------------------------------
# Approaching / receding targets
# ---------------------------------------------------------------------------


def test_approaching_target_has_positive_closing_velocity():
    state = GeometryEngine.compute_relative_state(
        "T1", 0.0, Vector3(0, 0, 0), ZERO, 0.0, Vector3(100, 0, 0), Vector3(-10, 0, 0)
    )
    assert state.closing_velocity_mps == pytest.approx(10.0)


def test_receding_target_has_negative_closing_velocity():
    state = GeometryEngine.compute_relative_state(
        "T1", 0.0, Vector3(0, 0, 0), ZERO, 0.0, Vector3(100, 0, 0), Vector3(10, 0, 0)
    )
    assert state.closing_velocity_mps == pytest.approx(-10.0)


def test_crossing_target_has_zero_closing_velocity():
    """Target moving perpendicular to the line-of-sight is neither
    approaching nor receding."""
    state = GeometryEngine.compute_relative_state(
        "T1", 0.0, Vector3(0, 0, 0), ZERO, 0.0, Vector3(100, 0, 0), Vector3(0, 10, 0)
    )
    assert state.closing_velocity_mps == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Zero velocity
# ---------------------------------------------------------------------------


def test_zero_velocity_gives_zero_closing_velocity():
    state = GeometryEngine.compute_relative_state(
        "T1", 0.0, Vector3(1, 2, 3), ZERO, 0.0, Vector3(100, 50, 25), ZERO
    )
    assert state.relative_velocity == Vector3(0, 0, 0)
    assert state.closing_velocity_mps == 0.0


# ---------------------------------------------------------------------------
# Moving ownship / moving target / moving both
# ---------------------------------------------------------------------------


def test_moving_ownship_only_toward_target():
    state = GeometryEngine.compute_relative_state(
        "T1", 0.0, Vector3(0, 0, 0), Vector3(10, 0, 0), 0.0, Vector3(100, 0, 0), ZERO
    )
    assert state.closing_velocity_mps == pytest.approx(10.0)


def test_moving_target_only_away_from_ownship():
    state = GeometryEngine.compute_relative_state(
        "T1", 0.0, Vector3(0, 0, 0), ZERO, 0.0, Vector3(100, 0, 0), Vector3(10, 0, 0)
    )
    assert state.closing_velocity_mps == pytest.approx(-10.0)


def test_moving_both_ownship_and_target_toward_each_other():
    state = GeometryEngine.compute_relative_state(
        "T1", 0.0, Vector3(0, 0, 0), Vector3(5, 0, 0), 0.0, Vector3(100, 0, 0), Vector3(-5, 0, 0)
    )
    assert state.closing_velocity_mps == pytest.approx(10.0)


def test_moving_both_ownship_and_target_apart():
    state = GeometryEngine.compute_relative_state(
        "T1", 0.0, Vector3(0, 0, 0), Vector3(-5, 0, 0), 0.0, Vector3(100, 0, 0), Vector3(5, 0, 0)
    )
    assert state.closing_velocity_mps == pytest.approx(-10.0)


def test_moving_ownship_changes_relative_position_correctly():
    state = GeometryEngine.compute_relative_state(
        "T1", 1.0, Vector3(10, 0, 0), Vector3(1, 0, 0), 0.0, Vector3(100, 0, 0), ZERO
    )
    assert state.relative_position == Vector3(90, 0, 0)


# ---------------------------------------------------------------------------
# Ownship heading affects Bearing but not Azimuth
# ---------------------------------------------------------------------------


def test_bearing_accounts_for_ownship_heading():
    # Target due East; Ownship heading East (90) means target is dead ahead.
    state = GeometryEngine.compute_relative_state(
        "T1", 0.0, Vector3(0, 0, 0), ZERO, 90.0, Vector3(100, 0, 0), ZERO
    )
    assert state.bearing_deg == pytest.approx(0.0)
    # Azimuth is unaffected by ownship heading.
    assert state.azimuth_deg == pytest.approx(0.0)


def test_bearing_wraps_within_0_360():
    state = GeometryEngine.compute_relative_state(
        "T1", 0.0, Vector3(0, 0, 0), ZERO, 300.0, Vector3(100, 0, 0), ZERO
    )
    assert 0.0 <= state.bearing_deg < 360.0


# ---------------------------------------------------------------------------
# Negative coordinates
# ---------------------------------------------------------------------------


def test_negative_coordinates_matches_reference():
    ownship_position = Vector3(-50, -50, -10)
    ownship_velocity = Vector3(-2, 3, 0)
    ownship_heading = 45.0
    target_position = Vector3(-200, -10, -30)
    target_velocity = Vector3(4, -1, 2)

    state = GeometryEngine.compute_relative_state(
        "T1", 0.0, ownship_position, ownship_velocity, ownship_heading, target_position, target_velocity
    )
    reference = _reference_relative_state(
        ownship_position, ownship_velocity, ownship_heading, target_position, target_velocity
    )
    _assert_matches_reference(state, reference)


# ---------------------------------------------------------------------------
# Large coordinates
# ---------------------------------------------------------------------------


def test_large_coordinates_matches_reference_and_stays_finite():
    ownship_position = Vector3(1_000_000.0, -2_000_000.0, 50_000.0)
    ownship_velocity = Vector3(250.0, -100.0, 5.0)
    ownship_heading = 123.0
    target_position = Vector3(-5_000_000.0, 3_000_000.0, -20_000.0)
    target_velocity = Vector3(-300.0, 200.0, -10.0)

    state = GeometryEngine.compute_relative_state(
        "T1", 0.0, ownship_position, ownship_velocity, ownship_heading, target_position, target_velocity
    )
    reference = _reference_relative_state(
        ownship_position, ownship_velocity, ownship_heading, target_position, target_velocity
    )
    _assert_matches_reference(state, reference)

    assert math.isfinite(state.range_m)
    assert math.isfinite(state.azimuth_deg)
    assert math.isfinite(state.elevation_deg)
    assert math.isfinite(state.bearing_deg)
    assert math.isfinite(state.closing_velocity_mps)


# ---------------------------------------------------------------------------
# RelativeState is immutable
# ---------------------------------------------------------------------------


def test_relative_state_is_immutable():
    state = GeometryEngine.compute_relative_state(
        "T1", 0.0, Vector3(0, 0, 0), ZERO, 0.0, Vector3(100, 0, 0), ZERO
    )
    with pytest.raises(Exception):
        state.range_m = 999.0


# ---------------------------------------------------------------------------
# compute_batch: consistency with compute_relative_state, and edge cases
# ---------------------------------------------------------------------------


_BATCH_CASES = [
    # (ownship_pos, ownship_vel, heading, target_pos, target_vel)
    (Vector3(0, 0, 0), ZERO, 0.0, Vector3(0, 0, 0), ZERO),  # same position
    (Vector3(0, 0, 0), ZERO, 0.0, Vector3(100, 0, 0), ZERO),  # horizontal
    (Vector3(0, 0, 0), ZERO, 0.0, Vector3(0, 0, 100), ZERO),  # vertical
    (Vector3(0, 0, 0), ZERO, 0.0, Vector3(100, 100, 0), ZERO),  # 45 degree
    (Vector3(0, 0, 0), ZERO, 0.0, Vector3(100, 0, 0), Vector3(-10, 0, 0)),  # approaching
    (Vector3(0, 0, 0), ZERO, 0.0, Vector3(100, 0, 0), Vector3(10, 0, 0)),  # receding
    (Vector3(0, 0, 0), Vector3(5, 0, 0), 0.0, Vector3(100, 0, 0), Vector3(-5, 0, 0)),  # moving both
    (Vector3(-50, -50, -10), Vector3(-2, 3, 0), 45.0, Vector3(-200, -10, -30), Vector3(4, -1, 2)),  # negative
    (
        Vector3(1_000_000.0, -2_000_000.0, 50_000.0),
        Vector3(250.0, -100.0, 5.0),
        123.0,
        Vector3(-5_000_000.0, 3_000_000.0, -20_000.0),
        Vector3(-300.0, 200.0, -10.0),
    ),  # large
]


def test_compute_batch_matches_compute_relative_state_for_each_case():
    for ownship_position, ownship_velocity, heading, target_position, target_velocity in _BATCH_CASES:
        expected = GeometryEngine.compute_relative_state(
            "T1", 5.0, ownship_position, ownship_velocity, heading, target_position, target_velocity
        )
        [actual] = GeometryEngine.compute_batch(
            5.0, ownship_position, ownship_velocity, heading, [("T1", target_position, target_velocity)]
        )
        assert actual.range_m == pytest.approx(expected.range_m, abs=1e-6)
        assert actual.azimuth_deg == pytest.approx(expected.azimuth_deg, abs=1e-6)
        assert actual.elevation_deg == pytest.approx(expected.elevation_deg, abs=1e-6)
        assert actual.bearing_deg == pytest.approx(expected.bearing_deg, abs=1e-6)
        assert actual.closing_velocity_mps == pytest.approx(expected.closing_velocity_mps, abs=1e-6)


def test_compute_batch_handles_multiple_targets_in_order():
    results = GeometryEngine.compute_batch(
        0.0,
        Vector3(0, 0, 0),
        ZERO,
        0.0,
        [
            ("T1", Vector3(100, 0, 0), ZERO),
            ("T2", Vector3(0, 0, 100), ZERO),
            ("T3", Vector3(0, 0, 0), ZERO),
        ],
    )
    assert [r.target_id for r in results] == ["T1", "T2", "T3"]
    assert results[0].azimuth_deg == pytest.approx(0.0)
    assert results[1].elevation_deg == pytest.approx(90.0)
    assert results[2].range_m == 0.0


def test_compute_batch_empty_targets_returns_empty_list():
    assert GeometryEngine.compute_batch(0.0, Vector3(0, 0, 0), ZERO, 0.0, []) == []


def test_compute_batch_no_nan_across_mixed_zero_and_nonzero_ranges():
    results = GeometryEngine.compute_batch(
        0.0,
        Vector3(0, 0, 0),
        ZERO,
        0.0,
        [
            ("T1", Vector3(0, 0, 0), ZERO),  # zero range
            ("T2", Vector3(100, 0, 0), Vector3(-5, 0, 0)),  # nonzero range
        ],
    )
    for result in results:
        assert math.isfinite(result.range_m)
        assert math.isfinite(result.azimuth_deg)
        assert math.isfinite(result.elevation_deg)
        assert math.isfinite(result.bearing_deg)
        assert math.isfinite(result.closing_velocity_mps)
