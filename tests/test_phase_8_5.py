"""Tests for Phase 8.5 engineering refinements.

Covers: relative-geometry carry-through (Part 1), AuthenticationResult
(Part 2), expanded ReplyStatus/MeasurementStatus (Part 3), signal
strength monotonicity (Part 4, additional coverage beyond
test_propagation.py), track history (Part 5), track summary report
(Part 6), per-stage CSV logging enabled/disabled (Part 7), and
deterministic CSV generation / backward compatibility (Part 10).
"""

from __future__ import annotations

import csv as csv_module

import pytest

from iff_simulator.domain import Aircraft, AircraftState, Scenario, Vector3
from iff_simulator.sensors.iff import (
    AirborneTransponder,
    AuthenticationResult,
    IFFMode,
    IFFTrackManager,
    InterrogationMessage,
    InterrogationScheduler,
    MeasurementStatus,
    ModeDecoder,
    ReplyMatcher,
    ReplyStatus,
    TargetSelector,
    TrackStatus,
    UplinkFormat,
    compute_signal_strength,
    derive_authentication_status,
    write_decoded_csv,
    write_replies_csv,
    write_track_summary_csv,
    write_tracks_csv,
)
from iff_simulator.sensors.iff.measurement import DecodedIFFMeasurement
from iff_simulator.simulation import SimulationClock, World

ZERO = Vector3(0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Part 1: relative geometry carried through the pipeline
# ---------------------------------------------------------------------------


def _build_scenario_with_target(target_position=Vector3(100, 0, 0), target_velocity=Vector3(-5, 0, 0)):
    aircraft = [
        Aircraft(aircraft_id="OWNSHIP"),
        Aircraft(
            aircraft_id="T1",
            iff_capability="MODE_S_CAPABLE",
            mode_data={"enabled_modes": ["MODE_S"]},
        ),
    ]
    history = {
        "OWNSHIP": [AircraftState(time=1.0, position=ZERO, velocity=ZERO)],
        "T1": [AircraftState(time=1.0, position=target_position, velocity=target_velocity)],
    }
    return Scenario(aircraft, history)


def test_interrogation_message_carries_closing_velocity_and_relative_velocity():
    scenario = _build_scenario_with_target()
    clock = SimulationClock(start_time=1.0, dt=1.0, end_time=10.0)
    world = World(scenario, clock, ownship_id="OWNSHIP", maximum_range=1000.0, beam_width=360.0,
                  beam_height=180.0, interrogation_rate=1.0)
    selector = TargetSelector(world)
    scheduler = InterrogationScheduler(world, selector)

    interrogation = scheduler.tick()
    assert interrogation is not None
    # Target approaching at 5 m/s along -X, directly east of ownship: closing velocity should be positive.
    assert interrogation.closing_velocity_mps == pytest.approx(5.0)
    assert interrogation.relative_velocity == Vector3(-5, 0, 0)


def test_decoded_measurement_carries_relative_geometry_without_recomputation():
    scenario = _build_scenario_with_target()
    clock = SimulationClock(start_time=1.0, dt=1.0, end_time=10.0)
    world = World(scenario, clock, ownship_id="OWNSHIP", maximum_range=1000.0, beam_width=360.0,
                  beam_height=180.0, interrogation_rate=1.0)
    selector = TargetSelector(world)
    scheduler = InterrogationScheduler(world, selector)
    transponder = AirborneTransponder(scenario)
    matcher = ReplyMatcher()
    decoder = ModeDecoder()

    interrogation = scheduler.tick()
    reply = transponder.receive(interrogation)
    match_result = matcher.match(interrogation, reply, ZERO, Vector3(100, 0, 0))
    measurement = decoder.decode(match_result)

    # Geometry on the measurement is identical to what the interrogation
    # already carried -- no second GeometryEngine call happened.
    assert measurement.closing_velocity_mps == interrogation.closing_velocity_mps
    assert measurement.relative_velocity == interrogation.relative_velocity


def test_track_manager_receives_relative_velocity_automatically():
    scenario = _build_scenario_with_target()
    clock = SimulationClock(start_time=1.0, dt=1.0, end_time=10.0)
    world = World(scenario, clock, ownship_id="OWNSHIP", maximum_range=1000.0, beam_width=360.0,
                  beam_height=180.0, interrogation_rate=1.0)
    selector = TargetSelector(world)
    scheduler = InterrogationScheduler(world, selector)
    transponder = AirborneTransponder(scenario)
    matcher = ReplyMatcher()
    decoder = ModeDecoder()
    track_manager = IFFTrackManager()

    interrogation = scheduler.tick()
    reply = transponder.receive(interrogation)
    match_result = matcher.match(interrogation, reply, ZERO, Vector3(100, 0, 0))
    measurement = decoder.decode(match_result)

    # No relative_velocity override supplied -- IFFTrackManager must
    # still pick it up from the measurement itself.
    track = track_manager.update(measurement)
    assert track.relative_velocity == Vector3(-5, 0, 0)
    assert track.closing_velocity_mps == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Part 2: AuthenticationResult
# ---------------------------------------------------------------------------


def test_mode_s_authentication_status_is_not_applicable():
    assert derive_authentication_status(IFFMode.MODE_S, authenticated=False) == AuthenticationResult.NOT_APPLICABLE
    assert derive_authentication_status(IFFMode.MODE_S, authenticated=True) == AuthenticationResult.NOT_APPLICABLE


def test_mode5_success_is_authenticated():
    assert derive_authentication_status(IFFMode.MODE5_L1, authenticated=True) == AuthenticationResult.AUTHENTICATED
    assert derive_authentication_status(IFFMode.MODE5_L2, authenticated=True) == AuthenticationResult.AUTHENTICATED


def test_mode5_failure_is_failed():
    assert derive_authentication_status(IFFMode.MODE5_L1, authenticated=False) == AuthenticationResult.FAILED
    assert derive_authentication_status(IFFMode.MODE5_L2, authenticated=False) == AuthenticationResult.FAILED


def test_authentication_result_enum_has_exactly_three_members():
    assert {member.name for member in AuthenticationResult} == {"AUTHENTICATED", "FAILED", "NOT_APPLICABLE"}


# ---------------------------------------------------------------------------
# Part 3: expanded ReplyStatus / MeasurementStatus
# ---------------------------------------------------------------------------


def test_reply_status_original_values_unchanged():
    assert ReplyStatus.OK.value == "OK"
    assert ReplyStatus.FAILED_AUTH.value == "FAILED_AUTH"


def test_reply_status_has_new_placeholder_members():
    names = {member.name for member in ReplyStatus}
    for placeholder in ("VALID", "NO_REPLY", "TIMEOUT", "GARBLED", "FRUITED", "LATE_REPLY", "CRC_ERROR", "UNKNOWN_MODE"):
        assert placeholder in names


def test_measurement_status_original_values_unchanged():
    assert MeasurementStatus.VALID.value == "VALID"
    assert MeasurementStatus.NO_REPLY.value == "NO_REPLY"


def test_measurement_status_has_new_placeholder_members():
    names = {member.name for member in MeasurementStatus}
    for placeholder in ("TIMEOUT", "GARBLED", "FRUITED", "LATE_REPLY", "CRC_ERROR", "UNKNOWN_MODE"):
        assert placeholder in names


# ---------------------------------------------------------------------------
# Part 4: signal strength (additional coverage)
# ---------------------------------------------------------------------------


def test_signal_strength_never_negative_or_zero():
    for r in (0.0, 1e-9, 1.0, 1e12):
        assert compute_signal_strength(r) > 0.0


# ---------------------------------------------------------------------------
# Part 5: track history
# ---------------------------------------------------------------------------


def _measurement(target_id="T1", time=1.0, seq=1, status=MeasurementStatus.VALID, range_m=100.0):
    return DecodedIFFMeasurement(
        measurement_id=seq, time=time, target_id=target_id, ownship_id="OWNSHIP",
        mode=IFFMode.MODE_S, range_m=range_m, azimuth_deg=0.0, elevation_deg=0.0,
        icao_address="A00001" if status == MeasurementStatus.VALID else None,
        authentication_result=False, identity="UNKNOWN", mission=None,
        reply_status=status,
        processing_delay=50.0 if status == MeasurementStatus.VALID else None,
        propagation_delay=1.0 if status == MeasurementStatus.VALID else None,
        arrival_time=time if status == MeasurementStatus.VALID else None,
        sequence_number=seq,
    )


def test_track_history_bounded_at_20():
    manager = IFFTrackManager()
    for n in range(1, 26):  # 25 valid replies
        manager.update(_measurement(time=float(n), seq=n))
    history = manager.get_track_history("T1")
    assert len(history) == 20


def test_track_history_ordering_oldest_first():
    manager = IFFTrackManager()
    for n in range(1, 6):
        manager.update(_measurement(time=float(n), seq=n))
    history = manager.get_track_history("T1")
    assert [snap.sequence_number for snap in history] == [1, 2, 3, 4, 5]


def test_track_history_empty_for_unknown_aircraft():
    manager = IFFTrackManager()
    assert manager.get_track_history("NO_SUCH_AIRCRAFT") == []


# ---------------------------------------------------------------------------
# Part 6: track summary report
# ---------------------------------------------------------------------------


def test_completed_track_summary_correctness():
    manager = IFFTrackManager(miss_threshold=3, confirmation_threshold=2)
    manager.update(_measurement(time=1.0, seq=1, range_m=100.0))
    manager.update(_measurement(time=2.0, seq=2, range_m=200.0))  # confirmed here
    manager.update(_measurement(time=3.0, seq=3, status=MeasurementStatus.NO_REPLY))
    manager.update(_measurement(time=4.0, seq=4, status=MeasurementStatus.NO_REPLY))
    manager.update(_measurement(time=5.0, seq=5, status=MeasurementStatus.NO_REPLY))  # lost here

    summaries = manager.get_completed_track_summaries()
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.track_id == 1
    assert summary.aircraft_id == "T1"
    assert summary.track_start_time == 1.0
    assert summary.track_end_time == 5.0
    assert summary.duration == 4.0
    assert summary.replies_received == 2
    assert summary.replies_missed == 3
    assert summary.max_range_m == 200.0
    assert summary.min_range_m == 100.0
    assert summary.avg_range_m == 150.0
    assert summary.final_track_status == TrackStatus.LOST
    assert summary.tentative_time + summary.confirmed_time + summary.lost_time == pytest.approx(summary.duration)
    assert summary.lost_time == 0.0


def test_active_track_not_in_completed_summaries():
    manager = IFFTrackManager()
    manager.update(_measurement(time=1.0, seq=1))
    assert manager.get_completed_track_summaries() == []


def test_track_summary_csv_deterministic(tmp_path):
    manager = IFFTrackManager(miss_threshold=1)
    manager.update(_measurement(time=1.0, seq=1))
    manager.update(_measurement(time=2.0, seq=2, status=MeasurementStatus.NO_REPLY))
    summaries = manager.get_completed_track_summaries()

    path_a = write_track_summary_csv(summaries, tmp_path / "a.csv")
    path_b = write_track_summary_csv(summaries, tmp_path / "b.csv")
    assert path_a.read_text(encoding="utf-8") == path_b.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Part 7: logging enabled / disabled
# ---------------------------------------------------------------------------


def test_decoder_logging_disabled_by_default():
    decoder = ModeDecoder()
    assert decoder.enable_logging is False
    assert decoder.log == []


def test_decoder_logging_enabled_accumulates():
    decoder = ModeDecoder(enable_logging=True)
    scenario = _build_scenario_with_target()
    clock = SimulationClock(start_time=1.0, dt=1.0, end_time=3.0)
    world = World(scenario, clock, ownship_id="OWNSHIP", maximum_range=1000.0, beam_width=360.0,
                  beam_height=180.0, interrogation_rate=1.0)
    selector = TargetSelector(world)
    scheduler = InterrogationScheduler(world, selector)
    transponder = AirborneTransponder(scenario)
    matcher = ReplyMatcher()

    interrogation = scheduler.tick()
    reply = transponder.receive(interrogation)
    match_result = matcher.match(interrogation, reply, ZERO, Vector3(100, 0, 0))
    measurement = decoder.decode(match_result)

    assert decoder.log == [measurement]


def test_transponder_logging_disabled_produces_no_log():
    scenario = _build_scenario_with_target()
    transponder = AirborneTransponder(scenario)
    interrogation = InterrogationMessage(
        time=1.0, sequence_number=1, ownship_id="OWNSHIP", target_id="T1",
        mode=IFFMode.MODE_S, uplink_format=UplinkFormat.UF11, range_m=100.0, azimuth_deg=0.0, elevation_deg=0.0,
    )
    transponder.receive(interrogation)
    assert transponder.log == []


def test_transponder_logging_enabled_accumulates():
    scenario = _build_scenario_with_target()
    transponder = AirborneTransponder(scenario, enable_logging=True)
    interrogation = InterrogationMessage(
        time=1.0, sequence_number=1, ownship_id="OWNSHIP", target_id="T1",
        mode=IFFMode.MODE_S, uplink_format=UplinkFormat.UF11, range_m=100.0, azimuth_deg=0.0, elevation_deg=0.0,
    )
    reply = transponder.receive(interrogation)
    assert transponder.log == [reply]


def test_scheduler_logging_enabled_accumulates():
    scenario = _build_scenario_with_target()
    clock = SimulationClock(start_time=1.0, dt=1.0, end_time=3.0)
    world = World(scenario, clock, ownship_id="OWNSHIP", maximum_range=1000.0, beam_width=360.0,
                  beam_height=180.0, interrogation_rate=1.0)
    selector = TargetSelector(world)
    scheduler = InterrogationScheduler(world, selector, enable_logging=True)
    message = scheduler.tick()
    assert scheduler.log == [message]


def test_track_manager_logging_enabled_accumulates():
    manager = IFFTrackManager(enable_logging=True)
    track = manager.update(_measurement(time=1.0, seq=1))
    assert manager.log == [track]


def test_track_manager_logging_disabled_by_default():
    manager = IFFTrackManager()
    manager.update(_measurement(time=1.0, seq=1))
    assert manager.log == []


def test_enable_logging_does_not_change_pipeline_behavior():
    """The core success criterion of Part 7: logging must be purely additive."""
    m1 = _measurement(time=1.0, seq=1)
    m2 = _measurement(time=1.0, seq=1)

    manager_off = IFFTrackManager(enable_logging=False)
    manager_on = IFFTrackManager(enable_logging=True)

    track_off = manager_off.update(m1)
    track_on = manager_on.update(m2)

    assert track_off == track_on


# ---------------------------------------------------------------------------
# Part 10: deterministic CSV export for replies/decoded/tracks
# ---------------------------------------------------------------------------


def test_write_decoded_csv_deterministic(tmp_path):
    measurements = [_measurement(time=1.0, seq=1), _measurement(time=2.0, seq=2)]
    path_a = write_decoded_csv(measurements, tmp_path / "a.csv")
    path_b = write_decoded_csv(measurements, tmp_path / "b.csv")
    assert path_a.read_text(encoding="utf-8") == path_b.read_text(encoding="utf-8")


def test_write_tracks_csv_deterministic(tmp_path):
    manager = IFFTrackManager()
    track = manager.update(_measurement(time=1.0, seq=1))
    path_a = write_tracks_csv([track], tmp_path / "a.csv")
    path_b = write_tracks_csv([track], tmp_path / "b.csv")
    assert path_a.read_text(encoding="utf-8") == path_b.read_text(encoding="utf-8")


def test_write_replies_csv_deterministic(tmp_path):
    scenario = _build_scenario_with_target()
    transponder = AirborneTransponder(scenario)
    interrogation = InterrogationMessage(
        time=1.0, sequence_number=1, ownship_id="OWNSHIP", target_id="T1",
        mode=IFFMode.MODE_S, uplink_format=UplinkFormat.UF11, range_m=100.0, azimuth_deg=0.0, elevation_deg=0.0,
    )
    reply = transponder.receive(interrogation)
    path_a = write_replies_csv([reply], tmp_path / "a.csv")
    path_b = write_replies_csv([reply], tmp_path / "b.csv")
    assert path_a.read_text(encoding="utf-8") == path_b.read_text(encoding="utf-8")


def test_write_decoded_csv_readable_by_csv_reader(tmp_path):
    measurements = [_measurement(time=1.0, seq=1)]
    path = write_decoded_csv(measurements, tmp_path / "decoded.csv")
    with path.open(encoding="utf-8") as handle:
        rows = list(csv_module.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["Target_ID"] == "T1"
