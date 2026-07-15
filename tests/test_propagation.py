"""Tests for ReplyPropagation."""

from __future__ import annotations

import pytest

from iff_simulator.domain import Vector3
from iff_simulator.sensors.iff import (
    DEFAULT_REFERENCE_RANGE_M,
    SPEED_OF_LIGHT_MPS,
    IFFMode,
    ReplyMessage,
    ReplyPropagation,
    ReplyStatus,
    ReplyType,
    compute_signal_strength,
)

ZERO = Vector3(0.0, 0.0, 0.0)


def _reply(time: float = 10.0, processing_delay: float = 50.0) -> ReplyMessage:
    return ReplyMessage(
        reply_id=1,
        time=time,
        interrogation_sequence=1,
        ownship_id="OWNSHIP",
        target_id="T1",
        mode=IFFMode.MODE_S,
        reply_type=ReplyType.DF11,
        reply_status=ReplyStatus.OK,
        authenticated=False,
        mode_s_address="A00001",
        mode1=None,
        mode2=None,
        mode3A=None,
        modeC=None,
        mode5_level=None,
        payload=None,  # not needed for propagation math
        processing_delay=processing_delay,
    )


def test_propagation_delay_matches_distance_over_speed_of_light():
    propagation = ReplyPropagation()
    ownship_position = Vector3(0, 0, 0)
    target_position = Vector3(299_792_458.0, 0, 0)  # exactly 1 light-second away

    result = propagation.propagate(_reply(), ownship_position, target_position)

    assert result.propagation_delay_us == 1_000_000.0  # 1 second == 1,000,000 microseconds


def test_zero_distance_gives_zero_propagation_delay():
    propagation = ReplyPropagation()
    result = propagation.propagate(_reply(), Vector3(5, 5, 5), Vector3(5, 5, 5))
    assert result.propagation_delay_us == 0.0


def test_arrival_time_includes_processing_and_propagation_delay():
    propagation = ReplyPropagation()
    reply = _reply(time=100.0, processing_delay=50.0)
    # distance chosen so propagation_delay_us is a clean number: distance = speed_of_light * 25us
    distance = SPEED_OF_LIGHT_MPS * 25e-6
    result = propagation.propagate(reply, Vector3(0, 0, 0), Vector3(distance, 0, 0))

    assert result.propagation_delay_us == pytest.approx(25.0)
    expected_arrival = 100.0 + (50.0 + result.propagation_delay_us) / 1_000_000.0
    assert result.arrival_time == expected_arrival


# ---------------------------------------------------------------------------
# Phase 8.5 Part 4: deterministic inverse-square signal strength model
# ---------------------------------------------------------------------------


def test_signal_strength_at_zero_range_is_exactly_one():
    assert compute_signal_strength(0.0) == 1.0


def test_signal_strength_at_reference_range_is_one_half():
    assert compute_signal_strength(DEFAULT_REFERENCE_RANGE_M) == pytest.approx(0.5)


def test_signal_strength_approaches_zero_for_large_range():
    assert compute_signal_strength(1_000_000.0) == pytest.approx(0.0, abs=1e-5)


def test_signal_strength_is_always_positive_and_bounded():
    for range_m in (0.0, 1.0, 100.0, 1000.0, 1_000_000.0, 1e9):
        strength = compute_signal_strength(range_m)
        assert 0.0 < strength <= 1.0


def test_signal_strength_is_strictly_monotonically_decreasing():
    ranges = [0.0, 10.0, 100.0, 500.0, 1000.0, 5000.0, 50000.0]
    strengths = [compute_signal_strength(r) for r in ranges]
    for earlier, later in zip(strengths, strengths[1:]):
        assert later < earlier


def test_signal_strength_reference_range_is_configurable():
    assert compute_signal_strength(500.0, reference_range_m=500.0) == pytest.approx(0.5)


def test_signal_strength_varies_with_distance_in_propagate():
    propagation = ReplyPropagation()
    near = propagation.propagate(_reply(), Vector3(0, 0, 0), Vector3(1, 0, 0))
    far = propagation.propagate(_reply(), Vector3(0, 0, 0), Vector3(1_000_000, 0, 0))
    assert near.signal_strength > far.signal_strength
    assert near.signal_strength == pytest.approx(1.0, abs=1e-4)
    assert far.signal_strength == pytest.approx(0.0, abs=1e-5)


def test_propagate_signal_strength_uses_reference_range_override():
    propagation = ReplyPropagation(reference_range_m=100.0)
    result = propagation.propagate(_reply(), Vector3(0, 0, 0), Vector3(100.0, 0, 0))
    assert result.signal_strength == pytest.approx(0.5)


def test_propagate_is_deterministic():
    propagation = ReplyPropagation()
    reply = _reply()
    a = propagation.propagate(reply, Vector3(0, 0, 0), Vector3(500, 0, 0))
    b = propagation.propagate(reply, Vector3(0, 0, 0), Vector3(500, 0, 0))
    assert a == b


def test_reply_is_preserved_verbatim():
    propagation = ReplyPropagation()
    reply = _reply()
    result = propagation.propagate(reply, Vector3(0, 0, 0), Vector3(100, 0, 0))
    assert result.reply is reply


def test_custom_speed_of_light_is_injectable():
    propagation = ReplyPropagation(speed_of_light_mps=1000.0)
    result = propagation.propagate(_reply(), Vector3(0, 0, 0), Vector3(1000.0, 0, 0))
    # distance=1000m, speed=1000 m/s -> 1 second -> 1,000,000 microseconds
    assert result.propagation_delay_us == 1_000_000.0


def test_negative_and_large_coordinates_stay_finite():
    import math

    propagation = ReplyPropagation()
    result = propagation.propagate(
        _reply(), Vector3(-1_000_000, 2_000_000, -5000), Vector3(3_000_000, -4_000_000, 8000)
    )
    assert math.isfinite(result.propagation_delay_us)
    assert math.isfinite(result.arrival_time)
