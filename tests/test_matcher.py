"""Tests for ReplyMatcher."""

from __future__ import annotations

from iff_simulator.domain import Vector3
from iff_simulator.sensors.iff import (
    IFFMode,
    InterrogationMessage,
    ReplyMatcher,
    ReplyMessage,
    ReplyStatus,
    ReplyType,
    UplinkFormat,
)

ZERO = Vector3(0.0, 0.0, 0.0)


def _interrogation(
    mode: IFFMode = IFFMode.MODE_S, time: float = 100.0, sequence_number: int = 1,
    target_id: str = "T1", ownship_id: str = "OWNSHIP",
) -> InterrogationMessage:
    uplink_format = {
        IFFMode.MODE_S: UplinkFormat.UF11,
        IFFMode.MODE5_L1: UplinkFormat.UF20,
        IFFMode.MODE5_L2: UplinkFormat.UF21,
    }[mode]
    return InterrogationMessage(
        time=time, sequence_number=sequence_number, ownship_id=ownship_id, target_id=target_id,
        mode=mode, uplink_format=uplink_format, range_m=100.0, azimuth_deg=0.0, elevation_deg=0.0,
    )


def _reply(
    interrogation: InterrogationMessage, processing_delay: float = 50.0
) -> ReplyMessage:
    reply_type = {
        IFFMode.MODE_S: ReplyType.DF11, IFFMode.MODE5_L1: ReplyType.MODE5_L1_REPLY, IFFMode.MODE5_L2: ReplyType.MODE5_L2_REPLY,
    }[interrogation.mode]
    return ReplyMessage(
        reply_id=interrogation.sequence_number, time=interrogation.time,
        interrogation_sequence=interrogation.sequence_number, ownship_id=interrogation.ownship_id,
        target_id=interrogation.target_id, mode=interrogation.mode, reply_type=reply_type,
        reply_status=ReplyStatus.OK, authenticated=False, mode_s_address="A00001",
        mode1=None, mode2=None, mode3A=None, modeC=None, mode5_level=None,
        payload=None, processing_delay=processing_delay,
    )


def test_no_reply_at_all_is_timed_out():
    matcher = ReplyMatcher()
    interrogation = _interrogation()
    result = matcher.match(interrogation, None, Vector3(0, 0, 0), Vector3(100, 0, 0))
    assert result.timed_out is True
    assert result.propagated_reply is None


def test_matched_reply_within_timeout():
    matcher = ReplyMatcher()
    interrogation = _interrogation(mode=IFFMode.MODE_S)
    reply = _reply(interrogation)
    result = matcher.match(interrogation, reply, Vector3(0, 0, 0), Vector3(10, 0, 0))
    assert result.timed_out is False
    assert result.propagated_reply is not None
    assert result.propagated_reply.reply is reply


def test_matching_uses_sequence_target_mode_ownship():
    matcher = ReplyMatcher()
    interrogation = _interrogation(sequence_number=7, target_id="T9", ownship_id="OWN2")
    reply = _reply(interrogation)
    result = matcher.match(interrogation, reply, Vector3(0, 0, 0), Vector3(10, 0, 0))
    assert result.timed_out is False


def test_mismatched_sequence_is_treated_as_no_reply():
    matcher = ReplyMatcher()
    interrogation = _interrogation(sequence_number=1)
    reply = _reply(interrogation)
    mismatched_interrogation = _interrogation(sequence_number=2)  # different interrogation
    result = matcher.match(mismatched_interrogation, reply, Vector3(0, 0, 0), Vector3(10, 0, 0))
    assert result.timed_out is True
    assert result.propagated_reply is None


def test_mismatched_target_id_is_treated_as_no_reply():
    matcher = ReplyMatcher()
    interrogation = _interrogation(target_id="T1")
    reply = _reply(interrogation)
    other_interrogation = _interrogation(target_id="T2", sequence_number=interrogation.sequence_number)
    result = matcher.match(other_interrogation, reply, Vector3(0, 0, 0), Vector3(10, 0, 0))
    assert result.timed_out is True


def test_timeout_from_excessive_propagation_distance():
    """A target far enough away that light-speed propagation alone exceeds
    the mode's timeout window results in a timed-out match, even though a
    reply object exists."""
    matcher = ReplyMatcher()
    interrogation = _interrogation(mode=IFFMode.MODE_S, time=100.0)
    reply = _reply(interrogation, processing_delay=0.0)
    # Mode S timeout is 500us; choose a distance whose light-speed delay
    # alone exceeds that (e.g. 200 km -> ~667us).
    far_position = Vector3(200_000.0, 0.0, 0.0)
    result = matcher.match(interrogation, reply, Vector3(0, 0, 0), far_position)
    assert result.timed_out is True
    assert result.propagated_reply is None


def test_within_timeout_for_realistic_short_range():
    matcher = ReplyMatcher()
    interrogation = _interrogation(mode=IFFMode.MODE_S, time=100.0)
    reply = _reply(interrogation, processing_delay=50.0)
    result = matcher.match(interrogation, reply, Vector3(0, 0, 0), Vector3(1000.0, 0, 0))
    assert result.timed_out is False


def test_receiver_actually_buffers_matched_replies():
    matcher = ReplyMatcher()
    interrogation = _interrogation()
    reply = _reply(interrogation)
    matcher.match(interrogation, reply, Vector3(0, 0, 0), Vector3(10, 0, 0))
    assert len(matcher.receiver) == 1


def test_match_is_deterministic():
    matcher = ReplyMatcher()
    interrogation = _interrogation()
    reply = _reply(interrogation)
    result_a = matcher.match(interrogation, reply, Vector3(0, 0, 0), Vector3(10, 0, 0))

    matcher2 = ReplyMatcher()
    result_b = matcher2.match(interrogation, reply, Vector3(0, 0, 0), Vector3(10, 0, 0))

    assert result_a.timed_out == result_b.timed_out
    assert result_a.propagated_reply.arrival_time == result_b.propagated_reply.arrival_time
