"""Exhaustive tests for IFFTrackManager."""

from __future__ import annotations

import pytest

from iff_simulator.domain import Vector3
from iff_simulator.sensors.iff import (
    FriendFoeStatus,
    IFFMode,
    IFFTrackManager,
    MeasurementStatus,
    TrackStatus,
)
from iff_simulator.sensors.iff.measurement import DecodedIFFMeasurement

ZERO = Vector3(0.0, 0.0, 0.0)


def _measurement(
    target_id: str = "T1",
    time: float = 1.0,
    sequence_number: int = 1,
    mode: IFFMode = IFFMode.MODE_S,
    reply_status: MeasurementStatus = MeasurementStatus.VALID,
    range_m: float = 100.0,
    azimuth_deg: float = 0.0,
    elevation_deg: float = 0.0,
    icao_address: str | None = "A00001",
    authentication_result: bool = False,
    identity: str = "UNKNOWN",
    mission: str | None = None,
    processing_delay: float | None = 50.0,
    propagation_delay: float | None = 1.0,
    arrival_time: float | None = None,
) -> DecodedIFFMeasurement:
    return DecodedIFFMeasurement(
        measurement_id=sequence_number,
        time=time,
        target_id=target_id,
        ownship_id="OWNSHIP",
        mode=mode,
        range_m=range_m,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        icao_address=icao_address,
        authentication_result=authentication_result,
        identity=identity,
        mission=mission,
        reply_status=reply_status,
        processing_delay=processing_delay,
        propagation_delay=propagation_delay,
        arrival_time=arrival_time if arrival_time is not None else time,
        sequence_number=sequence_number,
    )


def _no_reply(target_id: str = "T1", time: float = 1.0, sequence_number: int = 1) -> DecodedIFFMeasurement:
    return DecodedIFFMeasurement(
        measurement_id=sequence_number,
        time=time,
        target_id=target_id,
        ownship_id="OWNSHIP",
        mode=IFFMode.MODE_S,
        range_m=100.0,
        azimuth_deg=0.0,
        elevation_deg=0.0,
        icao_address=None,
        authentication_result=False,
        identity="UNKNOWN",
        mission=None,
        reply_status=MeasurementStatus.NO_REPLY,
        processing_delay=None,
        propagation_delay=None,
        arrival_time=None,
        sequence_number=sequence_number,
    )


# ---------------------------------------------------------------------------
# Track initiation
# ---------------------------------------------------------------------------


def test_first_valid_reply_initiates_tentative_track():
    manager = IFFTrackManager()
    track = manager.update(_measurement())
    assert track is not None
    assert track.track_status == TrackStatus.TENTATIVE
    assert track.track_quality == 0.3
    assert track.confidence == 0.3
    assert track.track_id == 1


def test_no_reply_for_unseen_aircraft_creates_no_track():
    manager = IFFTrackManager()
    track = manager.update(_no_reply())
    assert track is None
    assert manager.get_active_tracks() == []


def test_track_ids_are_assigned_sequentially_per_new_aircraft():
    manager = IFFTrackManager()
    track_a = manager.update(_measurement(target_id="A"))
    track_b = manager.update(_measurement(target_id="B"))
    assert track_a.track_id == 1
    assert track_b.track_id == 2


# ---------------------------------------------------------------------------
# Track confirmation
# ---------------------------------------------------------------------------


def test_confirms_after_3_consecutive_valid_replies():
    manager = IFFTrackManager()
    manager.update(_measurement(sequence_number=1, time=1.0))
    manager.update(_measurement(sequence_number=2, time=2.0))
    track = manager.update(_measurement(sequence_number=3, time=3.0))
    assert track.track_status == TrackStatus.CONFIRMED
    assert track.track_quality == 1.0
    assert track.confidence == 1.0


def test_does_not_confirm_after_only_2_valid_replies():
    manager = IFFTrackManager()
    manager.update(_measurement(sequence_number=1, time=1.0))
    track = manager.update(_measurement(sequence_number=2, time=2.0))
    assert track.track_status == TrackStatus.TENTATIVE
    assert track.track_quality == 0.3


def test_miss_resets_the_confirmation_streak():
    manager = IFFTrackManager()
    manager.update(_measurement(sequence_number=1, time=1.0))
    manager.update(_measurement(sequence_number=2, time=2.0))
    manager.update(_no_reply(sequence_number=3, time=3.0))  # breaks the streak
    manager.update(_measurement(sequence_number=4, time=4.0))
    manager.update(_measurement(sequence_number=5, time=5.0))
    track = manager.update(_measurement(sequence_number=6, time=6.0))
    # Only 3 consecutive valid replies since the miss (4,5,6) -> now confirmed.
    assert track.track_status == TrackStatus.CONFIRMED


def test_confirmation_threshold_is_configurable():
    manager = IFFTrackManager(confirmation_threshold=1)
    track = manager.update(_measurement())
    assert track.track_status == TrackStatus.CONFIRMED
    assert track.track_quality == 1.0


# ---------------------------------------------------------------------------
# Track update (fields updated on valid reply)
# ---------------------------------------------------------------------------


def test_valid_reply_updates_geometry_and_identity_fields():
    manager = IFFTrackManager()
    manager.update(_measurement(range_m=100.0, azimuth_deg=1.0, elevation_deg=2.0))
    track = manager.update(
        _measurement(sequence_number=2, time=2.0, range_m=200.0, azimuth_deg=5.0, elevation_deg=6.0, icao_address="A00099")
    )
    assert track.range_m == 200.0
    assert track.azimuth_deg == 5.0
    assert track.elevation_deg == 6.0
    assert track.mode_s_address == "A00099"


def test_valid_reply_updates_friend_foe_status():
    manager = IFFTrackManager()
    manager.update(_measurement(mode=IFFMode.MODE5_L1, authentication_result=False))
    track = manager.update(
        _measurement(sequence_number=2, time=2.0, mode=IFFMode.MODE5_L1, authentication_result=True)
    )
    assert track.friend_foe_status == FriendFoeStatus.FRIENDLY
    assert track.authentication_result is True


def test_valid_reply_updates_last_update_time_and_sequence_number():
    manager = IFFTrackManager()
    manager.update(_measurement(sequence_number=1, time=1.0))
    track = manager.update(_measurement(sequence_number=2, time=2.0))
    assert track.last_update_time == 2.0
    assert track.sequence_number == 2


def test_valid_reply_passes_through_optional_context():
    manager = IFFTrackManager()
    velocity = Vector3(1, 2, 3)
    track = manager.update(_measurement(), relative_velocity=velocity, signal_strength=0.5)
    assert track.relative_velocity == velocity
    assert track.signal_strength == 0.5


def test_relative_velocity_and_reply_type_default_to_none():
    manager = IFFTrackManager()
    track = manager.update(_measurement())
    assert track.relative_velocity is None
    assert track.reply_type is None


# ---------------------------------------------------------------------------
# Miss counter / quality penalty
# ---------------------------------------------------------------------------


def test_miss_increments_miss_count_and_reduces_quality():
    manager = IFFTrackManager()
    manager.update(_measurement(sequence_number=1, time=1.0))  # quality 0.3
    track = manager.update(_no_reply(sequence_number=2, time=2.0))
    assert track.track_quality == pytest.approx(0.2)


def test_quality_never_goes_below_zero():
    manager = IFFTrackManager(miss_threshold=100)  # keep the track alive through many misses
    manager.update(_measurement(sequence_number=1, time=1.0))  # 0.3
    for n in range(2, 10):
        track = manager.update(_no_reply(sequence_number=n, time=float(n)))
    assert track.track_quality == 0.0


def test_miss_does_not_update_geometry_mode_or_identity():
    manager = IFFTrackManager()
    manager.update(_measurement(range_m=111.0, azimuth_deg=1.0, elevation_deg=2.0, icao_address="A00001"))
    track = manager.update(_no_reply(sequence_number=2, time=2.0))
    assert track.range_m == 111.0
    assert track.azimuth_deg == 1.0
    assert track.elevation_deg == 2.0
    assert track.mode_s_address == "A00001"


def test_miss_sets_friend_foe_to_unknown():
    manager = IFFTrackManager()
    manager.update(_measurement(mode=IFFMode.MODE5_L1, authentication_result=True))  # FRIENDLY
    track = manager.update(_no_reply(sequence_number=2, time=2.0))
    assert track.friend_foe_status == FriendFoeStatus.UNKNOWN


def test_miss_does_not_advance_last_update_time_but_advances_time():
    manager = IFFTrackManager()
    manager.update(_measurement(sequence_number=1, time=1.0))
    track = manager.update(_no_reply(sequence_number=2, time=2.0))
    assert track.time == 2.0
    assert track.last_update_time == 1.0


def test_valid_reply_after_miss_resets_miss_count():
    manager = IFFTrackManager()
    manager.update(_measurement(sequence_number=1, time=1.0))  # 0.3
    manager.update(_no_reply(sequence_number=2, time=2.0))  # 0.2, miss_count=1
    track = manager.update(_measurement(sequence_number=3, time=3.0))  # valid -> quality back to 0.3 (still tentative)
    assert track.track_quality == 0.3


# ---------------------------------------------------------------------------
# Track deletion
# ---------------------------------------------------------------------------


def test_track_lost_after_miss_threshold_reached():
    manager = IFFTrackManager(miss_threshold=3)
    manager.update(_measurement(sequence_number=1, time=1.0))
    manager.update(_no_reply(sequence_number=2, time=2.0))
    manager.update(_no_reply(sequence_number=3, time=3.0))
    track = manager.update(_no_reply(sequence_number=4, time=4.0))
    assert track.track_status == TrackStatus.LOST


def test_lost_track_removed_from_active_tracks():
    manager = IFFTrackManager(miss_threshold=2)
    manager.update(_measurement(sequence_number=1, time=1.0))
    manager.update(_no_reply(sequence_number=2, time=2.0))
    manager.update(_no_reply(sequence_number=3, time=3.0))
    assert manager.get_active_tracks() == []
    assert manager.get_track("T1") is None


def test_miss_threshold_is_configurable():
    manager = IFFTrackManager(miss_threshold=1)
    manager.update(_measurement(sequence_number=1, time=1.0))
    track = manager.update(_no_reply(sequence_number=2, time=2.0))
    assert track.track_status == TrackStatus.LOST


def test_invalid_miss_threshold_raises():
    with pytest.raises(ValueError):
        IFFTrackManager(miss_threshold=0)


def test_invalid_confirmation_threshold_raises():
    with pytest.raises(ValueError):
        IFFTrackManager(confirmation_threshold=0)


def test_reinitiation_after_loss_gets_a_new_track_id():
    manager = IFFTrackManager(miss_threshold=1)
    first = manager.update(_measurement(sequence_number=1, time=1.0))
    manager.update(_no_reply(sequence_number=2, time=2.0))  # lost, track_id=1 removed
    assert manager.get_track("T1") is None
    second = manager.update(_measurement(sequence_number=3, time=3.0))
    assert second.track_id != first.track_id
    assert second.track_status == TrackStatus.TENTATIVE


# ---------------------------------------------------------------------------
# Association by Aircraft_ID (multiple independent tracks)
# ---------------------------------------------------------------------------


def test_multiple_aircraft_tracked_independently():
    manager = IFFTrackManager()
    manager.update(_measurement(target_id="A", sequence_number=1, time=1.0))
    manager.update(_measurement(target_id="B", sequence_number=1, time=1.0))
    manager.update(_no_reply(target_id="A", sequence_number=2, time=2.0))
    track_a = manager.get_track("A")
    track_b = manager.get_track("B")
    assert track_a.track_quality == pytest.approx(0.2)  # missed once
    assert track_b.track_quality == 0.3  # unaffected by A's miss


def test_get_active_tracks_returns_all_current_tracks():
    manager = IFFTrackManager()
    manager.update(_measurement(target_id="A"))
    manager.update(_measurement(target_id="B"))
    active_ids = {t.aircraft_id for t in manager.get_active_tracks()}
    assert active_ids == {"A", "B"}


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_measurement_sequence_produces_same_track_sequence():
    def run():
        manager = IFFTrackManager()
        results = []
        for n in range(1, 5):
            results.append(manager.update(_measurement(sequence_number=n, time=float(n))))
        return results

    first_run = run()
    second_run = run()
    assert [t.track_status for t in first_run] == [t.track_status for t in second_run]
    assert [t.track_quality for t in first_run] == [t.track_quality for t in second_run]
