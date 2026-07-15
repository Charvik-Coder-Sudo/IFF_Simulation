"""Tests for ReceiverBuffer."""

from __future__ import annotations

import random

import pytest

from iff_simulator.domain import Vector3
from iff_simulator.sensors.iff import (
    IFFMode,
    PropagatedReply,
    ReceiverBuffer,
    ReplyMessage,
    ReplyStatus,
    ReplyType,
)

ZERO = Vector3(0.0, 0.0, 0.0)


def _propagated(sequence_number: int, arrival_time: float) -> PropagatedReply:
    reply = ReplyMessage(
        reply_id=sequence_number,
        time=arrival_time,
        interrogation_sequence=sequence_number,
        ownship_id="OWNSHIP",
        target_id=f"T{sequence_number}",
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
        payload=None,
        processing_delay=50.0,
    )
    return PropagatedReply(reply=reply, arrival_time=arrival_time, propagation_delay_us=1.0, signal_strength=1.0)


def test_empty_buffer_length_and_peek():
    buffer = ReceiverBuffer()
    assert len(buffer) == 0
    assert buffer.peek() is None


def test_empty_buffer_pop_raises_index_error():
    buffer = ReceiverBuffer()
    with pytest.raises(IndexError):
        buffer.pop()


def test_insert_increases_length():
    buffer = ReceiverBuffer()
    buffer.insert(_propagated(1, 1.0))
    assert len(buffer) == 1


def test_ordered_by_arrival_time():
    buffer = ReceiverBuffer()
    buffer.insert(_propagated(3, 30.0))
    buffer.insert(_propagated(1, 10.0))
    buffer.insert(_propagated(2, 20.0))

    assert buffer.pop().arrival_time == 10.0
    assert buffer.pop().arrival_time == 20.0
    assert buffer.pop().arrival_time == 30.0


def test_ties_on_arrival_time_broken_by_sequence_number():
    buffer = ReceiverBuffer()
    buffer.insert(_propagated(3, 10.0))
    buffer.insert(_propagated(1, 10.0))
    buffer.insert(_propagated(2, 10.0))

    assert buffer.pop().reply.reply_id == 1
    assert buffer.pop().reply.reply_id == 2
    assert buffer.pop().reply.reply_id == 3


def test_peek_does_not_remove():
    buffer = ReceiverBuffer()
    buffer.insert(_propagated(1, 10.0))
    assert buffer.peek().reply.reply_id == 1
    assert len(buffer) == 1
    assert buffer.peek().reply.reply_id == 1


def test_clear_empties_buffer():
    buffer = ReceiverBuffer()
    buffer.insert(_propagated(1, 10.0))
    buffer.insert(_propagated(2, 20.0))
    buffer.clear()
    assert len(buffer) == 0
    assert buffer.peek() is None


def test_no_packet_loss_large_scenario():
    buffer = ReceiverBuffer()
    count = 5000
    order = list(range(count))
    random.Random(42).shuffle(order)

    for sequence_number in order:
        buffer.insert(_propagated(sequence_number, float(sequence_number)))

    assert len(buffer) == count

    popped = [buffer.pop().reply.reply_id for _ in range(count)]
    assert popped == list(range(count))  # strict arrival order, nothing lost or duplicated
    assert len(buffer) == 0
