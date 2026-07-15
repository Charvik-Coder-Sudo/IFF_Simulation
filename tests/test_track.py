"""Tests for derive_friend_foe_status and IFFTrack."""

from __future__ import annotations

import pytest

from iff_simulator.domain import Vector3
from iff_simulator.sensors.iff import (
    FriendFoeStatus,
    IFFMode,
    IFFTrack,
    MeasurementStatus,
    ReplyType,
    TrackStatus,
    derive_friend_foe_status,
)

ZERO = Vector3(0.0, 0.0, 0.0)


def test_no_reply_is_always_unknown():
    assert derive_friend_foe_status(IFFMode.MODE_S, True, MeasurementStatus.NO_REPLY) == FriendFoeStatus.UNKNOWN
    assert derive_friend_foe_status(IFFMode.MODE5_L1, True, MeasurementStatus.NO_REPLY) == FriendFoeStatus.UNKNOWN


def test_authenticated_mode5_l1_is_friendly():
    assert derive_friend_foe_status(IFFMode.MODE5_L1, True, MeasurementStatus.VALID) == FriendFoeStatus.FRIENDLY


def test_authenticated_mode5_l2_is_friendly():
    assert derive_friend_foe_status(IFFMode.MODE5_L2, True, MeasurementStatus.VALID) == FriendFoeStatus.FRIENDLY


def test_unauthenticated_mode5_l1_is_suspect():
    assert derive_friend_foe_status(IFFMode.MODE5_L1, False, MeasurementStatus.VALID) == FriendFoeStatus.SUSPECT


def test_unauthenticated_mode5_l2_is_suspect():
    assert derive_friend_foe_status(IFFMode.MODE5_L2, False, MeasurementStatus.VALID) == FriendFoeStatus.SUSPECT


def test_mode_s_is_always_unknown_regardless_of_authentication_flag():
    assert derive_friend_foe_status(IFFMode.MODE_S, False, MeasurementStatus.VALID) == FriendFoeStatus.UNKNOWN
    assert derive_friend_foe_status(IFFMode.MODE_S, True, MeasurementStatus.VALID) == FriendFoeStatus.UNKNOWN


def test_derive_friend_foe_status_is_deterministic():
    args = (IFFMode.MODE5_L1, True, MeasurementStatus.VALID)
    assert derive_friend_foe_status(*args) == derive_friend_foe_status(*args)


def _track(**overrides) -> IFFTrack:
    fields = dict(
        track_id=1,
        aircraft_id="T1",
        ownship_id="OWNSHIP",
        time=1.0,
        range_m=100.0,
        azimuth_deg=0.0,
        elevation_deg=0.0,
        relative_velocity=None,
        mode=IFFMode.MODE_S,
        reply_status=MeasurementStatus.VALID,
        mode_s_address="A00001",
        authentication_result=False,
        friend_foe_status=FriendFoeStatus.UNKNOWN,
        track_status=TrackStatus.TENTATIVE,
        track_quality=0.3,
        last_update_time=1.0,
        sequence_number=1,
        reply_type=ReplyType.DF11,
        confidence=0.3,
        signal_strength=1.0,
        propagation_delay=1.0,
    )
    fields.update(overrides)
    return IFFTrack(**fields)


def test_iff_track_is_immutable():
    track = _track()
    with pytest.raises(Exception):
        track.track_quality = 1.0


def test_iff_track_holds_supplied_fields():
    track = _track(track_id=5, aircraft_id="T9")
    assert track.track_id == 5
    assert track.aircraft_id == "T9"
