"""Tests for Phase 9 Part 2: false-alarm reply generation.

Covers: structural correctness (unknown ICAO/identity/authentication,
random-but-seeded sequence numbers), fresh phantom target_ids per call,
synthetic geometry bounds, determinism, and the resulting
IFFTrackManager Tentative -> Lost lifecycle with zero IFFTrackManager
changes.
"""

from __future__ import annotations

import random

import pytest

from iff_simulator.domain import Vector3
from iff_simulator.sensors.iff import (
    FalseReplyGenerator,
    IFFMode,
    IFFTrackManager,
    MeasurementStatus,
    ReplyStatus,
    TrackStatus,
)
from iff_simulator.sensors.iff.mode_s import ModeSPayload


def test_false_reply_has_unknown_identity_and_authentication():
    generator = FalseReplyGenerator(random.Random(1), ownship_id="OWNSHIP", maximum_range_m=1000.0)
    false_reply = generator.generate(time=1.0)
    reply = false_reply.reply

    assert reply.authenticated is False
    assert isinstance(reply.payload, ModeSPayload)
    assert reply.payload.identity == "UNKNOWN"
    assert reply.payload.capability == "UNKNOWN"
    assert reply.mode == IFFMode.MODE_S
    assert reply.reply_status == ReplyStatus.OK


def test_false_reply_target_id_is_phantom_and_fresh_each_call():
    generator = FalseReplyGenerator(random.Random(2), ownship_id="OWNSHIP", maximum_range_m=1000.0)
    first = generator.generate(time=1.0)
    second = generator.generate(time=2.0)
    assert first.reply.target_id.startswith("FALSE-")
    assert second.reply.target_id.startswith("FALSE-")
    assert first.reply.target_id != second.reply.target_id


def test_false_reply_sequence_number_is_seeded_random_not_fixed():
    generator_a = FalseReplyGenerator(random.Random(3), ownship_id="OWNSHIP", maximum_range_m=1000.0)
    generator_b = FalseReplyGenerator(random.Random(4), ownship_id="OWNSHIP", maximum_range_m=1000.0)
    reply_a = generator_a.generate(time=1.0).reply
    reply_b = generator_b.generate(time=1.0).reply
    assert reply_a.reply_id != reply_b.reply_id


def test_false_reply_generation_is_deterministic_given_same_seed():
    a = FalseReplyGenerator(random.Random(99), ownship_id="OWNSHIP", maximum_range_m=1000.0).generate(1.0)
    b = FalseReplyGenerator(random.Random(99), ownship_id="OWNSHIP", maximum_range_m=1000.0).generate(1.0)
    assert a.reply.reply_id == b.reply.reply_id
    assert a.reply.mode_s_address == b.reply.mode_s_address
    assert a.range_m == b.range_m
    assert a.azimuth_deg == b.azimuth_deg
    assert a.elevation_deg == b.elevation_deg


def test_false_reply_icao_address_is_six_hex_characters():
    generator = FalseReplyGenerator(random.Random(8), ownship_id="OWNSHIP", maximum_range_m=1000.0)
    for _ in range(20):
        false_reply = generator.generate(time=1.0)
        icao = false_reply.reply.mode_s_address
        assert len(icao) == 6
        assert all(c in "0123456789ABCDEF" for c in icao)


def test_false_reply_synthetic_geometry_within_configured_envelope():
    generator = FalseReplyGenerator(random.Random(5), ownship_id="OWNSHIP", maximum_range_m=1000.0)
    for _ in range(50):
        false_reply = generator.generate(time=1.0)
        assert 0.0 <= false_reply.range_m <= 1000.0
        assert 0.0 <= false_reply.azimuth_deg <= 360.0
        assert isinstance(false_reply.target_position, Vector3)


def test_false_alarm_measurement_creates_tentative_track_and_it_is_lost_without_followup():
    """No IFFTrackManager code changes were made for this -- a false
    alarm just looks like an ordinary VALID measurement for a phantom
    aircraft_id, and IFFTrackManager's existing miss-threshold logic
    ages it to Lost exactly like any other intermittent target."""
    manager = IFFTrackManager(miss_threshold=2, confirmation_threshold=3)
    generator = FalseReplyGenerator(random.Random(6), ownship_id="OWNSHIP", maximum_range_m=1000.0)
    false_reply = generator.generate(time=1.0)
    reply = false_reply.reply

    from iff_simulator.sensors.iff.measurement import DecodedIFFMeasurement

    valid_measurement = DecodedIFFMeasurement(
        measurement_id=reply.reply_id, time=1.0, target_id=reply.target_id, ownship_id="OWNSHIP",
        mode=reply.mode, range_m=false_reply.range_m, azimuth_deg=false_reply.azimuth_deg,
        elevation_deg=false_reply.elevation_deg, icao_address=reply.mode_s_address,
        authentication_result=False, identity="UNKNOWN", mission=None,
        reply_status=MeasurementStatus.VALID, processing_delay=reply.processing_delay,
        propagation_delay=1.0, arrival_time=1.0, sequence_number=reply.reply_id,
    )
    track = manager.update(valid_measurement)
    assert track.track_status == TrackStatus.TENTATIVE

    def no_reply_at(t):
        return DecodedIFFMeasurement(
            measurement_id=0, time=t, target_id=reply.target_id, ownship_id="OWNSHIP",
            mode=reply.mode, range_m=0.0, azimuth_deg=0.0, elevation_deg=0.0,
            icao_address=None, authentication_result=False, identity="UNKNOWN", mission=None,
            reply_status=MeasurementStatus.NO_REPLY, processing_delay=None,
            propagation_delay=None, arrival_time=None, sequence_number=0,
        )

    manager.update(no_reply_at(2.0))
    track = manager.update(no_reply_at(3.0))
    assert track.track_status == TrackStatus.LOST
    assert manager.get_track(reply.target_id) is None
