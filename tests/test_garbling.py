"""Tests for Phase 9 Part 3: garbling detection.

Covers: pairs inside/outside/exactly-at the window boundary, a single
reply never garbles, non-adjacent-but-nearby chains, and zero window
never garbles.
"""

from __future__ import annotations

from dataclasses import dataclass

from iff_simulator.sensors.iff import detect_garbled


@dataclass
class _FakePropagated:
    """Minimal stand-in for PropagatedReply: detect_garbled only reads
    `.arrival_time`, so a full ReplyMessage/PropagatedReply isn't needed."""

    arrival_time: float


def test_empty_batch_never_garbles():
    assert detect_garbled([], garble_window_s=1.0) == set()


def test_single_reply_never_garbles():
    batch = [_FakePropagated(arrival_time=1.0)]
    assert detect_garbled(batch, garble_window_s=1.0) == set()


def test_zero_window_never_garbles():
    batch = [_FakePropagated(arrival_time=1.0), _FakePropagated(arrival_time=1.0)]
    assert detect_garbled(batch, garble_window_s=0.0) == set()


def test_pair_within_window_both_garbled():
    a = _FakePropagated(arrival_time=1.0000)
    b = _FakePropagated(arrival_time=1.0003)
    garbled = detect_garbled([a, b], garble_window_s=0.0005)
    assert id(a) in garbled
    assert id(b) in garbled


def test_pair_outside_window_not_garbled():
    a = _FakePropagated(arrival_time=1.0000)
    b = _FakePropagated(arrival_time=1.0010)
    garbled = detect_garbled([a, b], garble_window_s=0.0005)
    assert garbled == set()


def test_pair_exactly_at_window_boundary_not_garbled():
    """`< garble_window_s`, not `<=` -- a delta exactly equal to the
    window is the boundary case, deliberately not garbled. Uses
    exactly-representable binary floats (0.5, 1.0) so the delta is
    exactly equal to the window with no floating-point rounding risk."""
    a = _FakePropagated(arrival_time=1.0)
    b = _FakePropagated(arrival_time=1.5)
    garbled = detect_garbled([a, b], garble_window_s=0.5)
    assert garbled == set()


def test_three_replies_only_close_pair_garbled():
    a = _FakePropagated(arrival_time=1.0000)
    b = _FakePropagated(arrival_time=1.0003)  # close to a
    c = _FakePropagated(arrival_time=2.0000)  # far from b
    garbled = detect_garbled([a, b, c], garble_window_s=0.0005)
    assert garbled == {id(a), id(b)}


def test_chain_of_three_all_within_window_are_all_garbled():
    a = _FakePropagated(arrival_time=1.0000)
    b = _FakePropagated(arrival_time=1.0002)
    c = _FakePropagated(arrival_time=1.0004)
    garbled = detect_garbled([a, b, c], garble_window_s=0.0005)
    assert garbled == {id(a), id(b), id(c)}
