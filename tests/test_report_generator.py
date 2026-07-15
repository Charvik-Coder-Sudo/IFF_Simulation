"""Tests for ReportGenerator / IFFMeasurementReport."""

from __future__ import annotations

from iff_simulator.domain import Vector3
from iff_simulator.sensors.iff import (
    FriendFoeStatus,
    IFFMode,
    IFFTrack,
    MeasurementStatus,
    ReplyType,
    ReportGenerator,
    TrackStatus,
)

ZERO = Vector3(0.0, 0.0, 0.0)


def _track(**overrides) -> IFFTrack:
    fields = dict(
        track_id=1,
        aircraft_id="T1",
        ownship_id="OWNSHIP",
        time=10.0,
        range_m=150.0,
        azimuth_deg=15.0,
        elevation_deg=3.0,
        relative_velocity=Vector3(1, 2, 3),
        mode=IFFMode.MODE5_L1,
        reply_status=MeasurementStatus.VALID,
        mode_s_address=None,
        authentication_result=True,
        friend_foe_status=FriendFoeStatus.FRIENDLY,
        track_status=TrackStatus.CONFIRMED,
        track_quality=1.0,
        last_update_time=10.0,
        sequence_number=42,
        reply_type=ReplyType.MODE5_L1_REPLY,
        confidence=1.0,
        signal_strength=1.0,
        propagation_delay=2.5,
    )
    fields.update(overrides)
    return IFFTrack(**fields)


def test_generate_copies_every_report_field():
    track = _track()
    report = ReportGenerator().generate(track)
    assert report.time == track.time
    assert report.track_id == track.track_id
    assert report.aircraft_id == track.aircraft_id
    assert report.ownship_id == track.ownship_id
    assert report.range_m == track.range_m
    assert report.azimuth_deg == track.azimuth_deg
    assert report.elevation_deg == track.elevation_deg
    assert report.relative_velocity == track.relative_velocity
    assert report.mode == track.mode
    assert report.reply_status == track.reply_status
    assert report.mode_s_address == track.mode_s_address
    assert report.authentication_result == track.authentication_result
    assert report.friend_foe_status == track.friend_foe_status
    assert report.track_quality == track.track_quality
    assert report.track_status == track.track_status
    assert report.sequence_number == track.sequence_number
    assert report.signal_strength == track.signal_strength
    assert report.propagation_delay == track.propagation_delay


def test_report_excludes_last_update_time_reply_type_and_confidence():
    report_fields = set(ReportGenerator().generate(_track()).__dataclass_fields__.keys())
    assert "last_update_time" not in report_fields
    assert "reply_type" not in report_fields
    assert "confidence" not in report_fields


def test_report_has_exactly_18_fields():
    report = ReportGenerator().generate(_track())
    assert len(report.__dataclass_fields__) == 18


def test_generate_many_preserves_order():
    tracks = [_track(track_id=i, aircraft_id=f"T{i}") for i in (3, 1, 2)]
    reports = ReportGenerator().generate_many(tracks)
    assert [r.track_id for r in reports] == [3, 1, 2]
