"""Tests for Phase 9 Part 8: timing jitter.

Covers: zero jitter -> unchanged, jittered delay/arrival time stays
within the configured +/- bound over many seeded draws, and
`JitteredReplyPropagation` reproduces the base `ReplyPropagation`'s
distance/signal-strength computation exactly.
"""

from __future__ import annotations

import random

import pytest

from iff_simulator.domain import Vector3
from iff_simulator.sensors.iff import (
    IFFMode,
    JitteredReplyPropagation,
    ReplyMessage,
    ReplyPropagation,
    ReplyStatus,
    ReplyType,
    compute_signal_strength,
    jitter_processing_delay,
)

ZERO = Vector3(0.0, 0.0, 0.0)


def _reply(time: float = 10.0, processing_delay: float = 50.0) -> ReplyMessage:
    return ReplyMessage(
        reply_id=1, time=time, interrogation_sequence=1, ownship_id="OWNSHIP", target_id="T1",
        mode=IFFMode.MODE_S, reply_type=ReplyType.DF11, reply_status=ReplyStatus.OK,
        authenticated=False, mode_s_address="A00001", mode1=None, mode2=None, mode3A=None,
        modeC=None, mode5_level=None, payload=None, processing_delay=processing_delay,
    )


def test_zero_jitter_processing_delay_returns_same_reply():
    reply = _reply(processing_delay=50.0)
    jittered = jitter_processing_delay(reply, 0.0, random.Random(1))
    assert jittered is reply


def test_processing_delay_jitter_stays_within_bound():
    rng = random.Random(2)
    reply = _reply(processing_delay=50.0)
    for _ in range(2000):
        jittered = jitter_processing_delay(reply, 5.0, rng)
        delta = jittered.processing_delay - reply.processing_delay
        assert -5.0 <= delta <= 5.0


def test_zero_jitter_propagation_matches_base_reply_propagation():
    rng = random.Random(3)
    jittered_propagation = JitteredReplyPropagation(rng, jitter_us=0.0)
    base_propagation = ReplyPropagation()
    reply = _reply()
    ownship_position = ZERO
    target_position = Vector3(1000.0, 0.0, 0.0)

    result_a = jittered_propagation.propagate(reply, ownship_position, target_position)
    result_b = base_propagation.propagate(reply, ownship_position, target_position)

    assert result_a.propagation_delay_us == result_b.propagation_delay_us
    assert result_a.arrival_time == result_b.arrival_time
    assert result_a.signal_strength == result_b.signal_strength


def test_propagation_delay_jitter_stays_within_bound_and_updates_arrival_time():
    rng = random.Random(4)
    jitter_us = 10.0
    jittered_propagation = JitteredReplyPropagation(rng, jitter_us=jitter_us)
    reply = _reply()
    ownship_position = ZERO
    target_position = Vector3(1000.0, 0.0, 0.0)

    base_delay = ReplyPropagation().propagate(reply, ownship_position, target_position).propagation_delay_us

    for _ in range(2000):
        result = jittered_propagation.propagate(reply, ownship_position, target_position)
        delta_us = result.propagation_delay_us - base_delay
        assert -jitter_us <= delta_us <= jitter_us
        expected_arrival = reply.time + (reply.processing_delay + result.propagation_delay_us) / 1_000_000.0
        assert result.arrival_time == pytest.approx(expected_arrival)


def test_processing_delay_jitter_distribution_is_centered_on_zero():
    rng = random.Random(21)
    reply = _reply(processing_delay=50.0)
    deltas = [jitter_processing_delay(reply, 20.0, rng).processing_delay - 50.0 for _ in range(5000)]
    assert sum(deltas) / len(deltas) == pytest.approx(0.0, abs=1.0)


def test_negative_jitter_bound_would_be_symmetric_uniform_range():
    """rng.uniform(-jitter, jitter) is symmetric regardless of sign
    convention -- this just documents that a jitter_us of 0.0 is the
    only value that ever short-circuits to "no change"."""
    rng = random.Random(22)
    reply = _reply(processing_delay=50.0)
    jittered = jitter_processing_delay(reply, 1e-9, rng)
    assert jittered is not reply


def test_jittered_propagation_is_deterministic_given_same_seed():
    reply = _reply()
    ownship_position = ZERO
    target_position = Vector3(1000.0, 0.0, 0.0)

    a = JitteredReplyPropagation(random.Random(77), jitter_us=10.0).propagate(
        reply, ownship_position, target_position
    )
    b = JitteredReplyPropagation(random.Random(77), jitter_us=10.0).propagate(
        reply, ownship_position, target_position
    )
    assert a.propagation_delay_us == b.propagation_delay_us
    assert a.arrival_time == b.arrival_time
