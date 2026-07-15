"""Tests for Receiver."""

from __future__ import annotations

from iff_simulator.domain import Vector3
from iff_simulator.sensors.iff import (
    IFFMode,
    InterrogationMessage,
    PropagatedReply,
    Receiver,
    ReplyMessage,
    ReplyStatus,
    ReplyType,
    UplinkFormat,
)

ZERO = Vector3(0.0, 0.0, 0.0)


def _interrogation(mode: IFFMode, time: float = 100.0, sequence_number: int = 1) -> InterrogationMessage:
    uplink_format = {
        IFFMode.MODE_S: UplinkFormat.UF11,
        IFFMode.MODE5_L1: UplinkFormat.UF20,
        IFFMode.MODE5_L2: UplinkFormat.UF21,
    }[mode]
    return InterrogationMessage(
        time=time, sequence_number=sequence_number, ownship_id="OWNSHIP", target_id="T1",
        mode=mode, uplink_format=uplink_format, range_m=100.0, azimuth_deg=0.0, elevation_deg=0.0,
    )


def _propagated(sequence_number: int, arrival_time: float, mode: IFFMode = IFFMode.MODE_S) -> PropagatedReply:
    reply_type = {IFFMode.MODE_S: ReplyType.DF11, IFFMode.MODE5_L1: ReplyType.MODE5_L1_REPLY, IFFMode.MODE5_L2: ReplyType.MODE5_L2_REPLY}[mode]
    reply = ReplyMessage(
        reply_id=sequence_number, time=arrival_time, interrogation_sequence=sequence_number,
        ownship_id="OWNSHIP", target_id="T1", mode=mode, reply_type=reply_type,
        reply_status=ReplyStatus.OK, authenticated=False, mode_s_address="A00001",
        mode1=None, mode2=None, mode3A=None, modeC=None, mode5_level=None,
        payload=None, processing_delay=50.0,
    )
    return PropagatedReply(reply=reply, arrival_time=arrival_time, propagation_delay_us=1.0, signal_strength=1.0)


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


def test_mode_s_timeout_is_500_microseconds():
    interrogation = _interrogation(IFFMode.MODE_S, time=100.0)
    just_inside = 100.0 + 499e-6
    just_outside = 100.0 + 501e-6
    assert Receiver.is_timed_out(interrogation, just_inside) is False
    assert Receiver.is_timed_out(interrogation, just_outside) is True


def test_mode5_timeout_is_700_microseconds():
    interrogation = _interrogation(IFFMode.MODE5_L1, time=100.0)
    just_inside = 100.0 + 699e-6
    just_outside = 100.0 + 701e-6
    assert Receiver.is_timed_out(interrogation, just_inside) is False
    assert Receiver.is_timed_out(interrogation, just_outside) is True


def test_exactly_at_deadline_is_not_timed_out():
    """Boundary condition: exactly at the deadline is still on time (strict >)."""
    interrogation = _interrogation(IFFMode.MODE_S, time=100.0)
    deadline = 100.0 + 500e-6
    assert Receiver.is_timed_out(interrogation, deadline) is False


# ---------------------------------------------------------------------------
# Reply acceptance / ordering / forwarding
# ---------------------------------------------------------------------------


def test_receive_accepts_into_buffer():
    receiver = Receiver()
    receiver.receive(_propagated(1, 100.0001))
    assert len(receiver) == 1


def test_pop_ready_only_returns_arrived_replies():
    receiver = Receiver()
    receiver.receive(_propagated(1, 100.0005))  # arrives at 100.0005
    receiver.receive(_propagated(2, 200.0))  # not arrived yet as of current_time=100.001

    ready = receiver.pop_ready(current_time=100.001)
    assert [p.reply.reply_id for p in ready] == [1]
    assert len(receiver) == 1  # the not-yet-arrived one remains buffered


def test_pop_ready_returns_in_arrival_order():
    receiver = Receiver()
    receiver.receive(_propagated(3, 100.003))
    receiver.receive(_propagated(1, 100.001))
    receiver.receive(_propagated(2, 100.002))

    ready = receiver.pop_ready(current_time=1000.0)
    assert [p.reply.reply_id for p in ready] == [1, 2, 3]


def test_pop_ready_empty_when_nothing_arrived():
    receiver = Receiver()
    receiver.receive(_propagated(1, 500.0))
    assert receiver.pop_ready(current_time=1.0) == []
    assert len(receiver) == 1
