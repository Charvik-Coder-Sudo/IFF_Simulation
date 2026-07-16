"""Tests for Phase 10 statistics.py.

Covers: the small numeric helpers (mean/population_stdev/min_max/
safe_divide), per-mode DetectionStatistics rows, per-level
AuthenticationStatistics rows, TrackStatistics rows for both completed
and still-active tracks, and deterministic CSV output for all three.
"""

from __future__ import annotations

import pytest

from iff_simulator.domain import Aircraft, AircraftState, Scenario, Vector3
from iff_simulator.sensors.iff import IFFMode, MeasurementStatus, ReceiverTickResult, TrackStatus
from iff_simulator.sensors.iff.authentication import AuthenticationResult
from iff_simulator.sensors.iff.measurement import DecodedIFFMeasurement
from iff_simulator.sensors.iff.track import FriendFoeStatus, IFFTrack
from iff_simulator.sensors.iff.track_summary import TrackSummary
from iff_simulator.analysis import (
    PipelineRunRecord,
    compute_authentication_statistics,
    compute_detection_statistics,
    compute_track_statistics,
    mean,
    min_max,
    population_stdev,
    safe_divide,
    write_authentication_statistics_csv,
    write_detection_statistics_csv,
    write_track_statistics_csv,
)

ZERO = Vector3(0.0, 0.0, 0.0)


def _scenario():
    aircraft = [Aircraft(aircraft_id="OWNSHIP"), Aircraft(aircraft_id="T1")]
    history = {a.aircraft_id: [AircraftState(time=1.0, position=ZERO, velocity=ZERO)] for a in aircraft}
    return Scenario(aircraft, history)


def _interrogation_and_measurement(seq, mode, range_m=100.0, valid=True):
    from iff_simulator.sensors.iff import InterrogationMessage, UplinkFormat

    interrogation = InterrogationMessage(
        time=1.0, sequence_number=seq, ownship_id="OWNSHIP", target_id="T1", mode=mode,
        uplink_format=UplinkFormat.UF11, range_m=range_m, azimuth_deg=0.0, elevation_deg=0.0,
    )
    status = MeasurementStatus.VALID if valid else MeasurementStatus.NO_REPLY
    measurement = DecodedIFFMeasurement(
        measurement_id=seq, time=1.0, target_id="T1", ownship_id="OWNSHIP", mode=mode, range_m=range_m,
        azimuth_deg=0.0, elevation_deg=0.0, icao_address="A00001" if valid else None, authentication_result=False,
        identity="BLUE" if valid else "UNKNOWN", mission=None, reply_status=status,
        processing_delay=50.0 if valid else None, propagation_delay=1.0 if valid else None,
        arrival_time=1.0 if valid else None, sequence_number=seq, signal_strength=0.9 if valid else None,
    )
    return interrogation, measurement


def _tick(measurement):
    return ReceiverTickResult(real_measurement=measurement, false_alarm_measurements=[], fruited_measurements=[])


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def test_mean_and_min_max_and_stdev_on_empty_sequence():
    assert mean([]) == 0.0
    assert min_max([]) == (0.0, 0.0)
    assert population_stdev([]) == 0.0
    assert population_stdev([5.0]) == 0.0


def test_mean_and_min_max_and_stdev_on_values():
    values = [1.0, 2.0, 3.0]
    assert mean(values) == pytest.approx(2.0)
    assert min_max(values) == (1.0, 3.0)
    assert population_stdev(values) > 0.0


def test_safe_divide_never_raises_on_zero_denominator():
    assert safe_divide(5.0, 0.0) == 0.0
    assert safe_divide(0.0, 0.0) == 0.0
    assert safe_divide(6.0, 3.0) == 2.0


# ---------------------------------------------------------------------------
# Detection statistics
# ---------------------------------------------------------------------------


def test_detection_statistics_one_row_per_mode_plus_all():
    interrogation1, measurement1 = _interrogation_and_measurement(1, IFFMode.MODE_S, range_m=100.0)
    interrogation2, measurement2 = _interrogation_and_measurement(2, IFFMode.MODE_S, range_m=200.0, valid=False)
    record = PipelineRunRecord(
        scenario=_scenario(), interrogations=[interrogation1, interrogation2],
        replies=[object(), object()], tick_results=[_tick(measurement1), _tick(measurement2)],
    )
    rows = compute_detection_statistics(record)
    by_mode = {row.mode: row for row in rows}
    assert set(by_mode.keys()) == {"MODE_S", "MODE5_L1", "MODE5_L2", "ALL"}

    mode_s = by_mode["MODE_S"]
    assert mode_s.interrogations == 2
    assert mode_s.expected_replies == 2
    assert mode_s.correct_replies == 1
    assert mode_s.detection_probability == pytest.approx(0.5)
    assert mode_s.average_range_m == pytest.approx(100.0)
    assert mode_s.maximum_range_m == pytest.approx(100.0)

    all_row = by_mode["ALL"]
    assert all_row.interrogations == 2
    assert all_row.correct_replies == 1


def test_detection_statistics_empty_mode_gives_zero_not_crash():
    record = PipelineRunRecord(scenario=_scenario())
    rows = compute_detection_statistics(record)
    for row in rows:
        assert row.detection_probability == 0.0
        assert row.interrogations == 0


def test_write_detection_statistics_csv_deterministic(tmp_path):
    interrogation, measurement = _interrogation_and_measurement(1, IFFMode.MODE_S)
    record = PipelineRunRecord(
        scenario=_scenario(), interrogations=[interrogation], replies=[object()], tick_results=[_tick(measurement)],
    )
    rows = compute_detection_statistics(record)
    path_a = write_detection_statistics_csv(rows, tmp_path / "a.csv")
    path_b = write_detection_statistics_csv(rows, tmp_path / "b.csv")
    assert path_a.read_text(encoding="utf-8") == path_b.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Authentication statistics
# ---------------------------------------------------------------------------


def _mode5_measurement(seq, mode, auth_status):
    return DecodedIFFMeasurement(
        measurement_id=seq, time=1.0, target_id="T1", ownship_id="OWNSHIP", mode=mode, range_m=100.0,
        azimuth_deg=0.0, elevation_deg=0.0, icao_address=None,
        authentication_result=(auth_status == AuthenticationResult.AUTHENTICATED), identity="BLUE", mission=None,
        reply_status=MeasurementStatus.VALID, processing_delay=75.0, propagation_delay=1.0, arrival_time=1.0,
        sequence_number=seq, authentication_status=auth_status,
    )


def test_authentication_statistics_per_level_and_all_mode5():
    tick_results = [
        _tick(_mode5_measurement(1, IFFMode.MODE5_L1, AuthenticationResult.AUTHENTICATED)),
        _tick(_mode5_measurement(2, IFFMode.MODE5_L1, AuthenticationResult.FAILED)),
        _tick(_mode5_measurement(3, IFFMode.MODE5_L2, AuthenticationResult.AUTHENTICATED)),
    ]
    record = PipelineRunRecord(scenario=_scenario(), tick_results=tick_results)
    rows = compute_authentication_statistics(record)
    by_mode = {row.mode: row for row in rows}

    assert by_mode["MODE5_L1"].mode5_replies == 2
    assert by_mode["MODE5_L1"].authenticated == 1
    assert by_mode["MODE5_L1"].authentication_success_rate == pytest.approx(0.5)

    assert by_mode["MODE5_L2"].mode5_replies == 1
    assert by_mode["MODE5_L2"].authenticated == 1
    assert by_mode["MODE5_L2"].authentication_success_rate == pytest.approx(1.0)

    assert by_mode["ALL_MODE5"].mode5_replies == 3
    assert by_mode["ALL_MODE5"].authenticated == 2


def test_write_authentication_statistics_csv_deterministic(tmp_path):
    tick_results = [_tick(_mode5_measurement(1, IFFMode.MODE5_L1, AuthenticationResult.AUTHENTICATED))]
    record = PipelineRunRecord(scenario=_scenario(), tick_results=tick_results)
    rows = compute_authentication_statistics(record)
    path_a = write_authentication_statistics_csv(rows, tmp_path / "a.csv")
    path_b = write_authentication_statistics_csv(rows, tmp_path / "b.csv")
    assert path_a.read_text(encoding="utf-8") == path_b.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Track statistics
# ---------------------------------------------------------------------------


def test_track_statistics_completed_track_row():
    summary = TrackSummary(
        track_id=1, aircraft_id="T1", ownship_id="OWNSHIP", track_start_time=0.0, track_end_time=10.0,
        duration=10.0, replies_received=8, replies_missed=2, tentative_time=3.0, confirmed_time=7.0,
        lost_time=0.0, max_range_m=200.0, min_range_m=100.0, avg_range_m=150.0, max_signal_strength=0.9,
        avg_signal_strength=0.8, final_track_status=TrackStatus.LOST,
    )
    record = PipelineRunRecord(scenario=_scenario(), completed_track_summaries=[summary])
    rows = compute_track_statistics(record)
    assert len(rows) == 1
    row = rows[0]
    assert row.track_id == 1
    assert row.aircraft_id == "T1"
    assert row.track_status == "Lost"
    assert row.ever_confirmed is True
    assert row.duration == pytest.approx(10.0)
    assert row.replies_received == 8
    assert row.average_range_m == pytest.approx(150.0)


def test_track_statistics_active_track_row_has_partial_fields():
    track = IFFTrack(
        track_id=2, aircraft_id="T2", ownship_id="OWNSHIP", time=1.0, range_m=175.0, azimuth_deg=0.0,
        elevation_deg=0.0, relative_velocity=None, mode=IFFMode.MODE_S, reply_status=MeasurementStatus.VALID,
        mode_s_address="A00002", authentication_result=False, friend_foe_status=FriendFoeStatus.UNKNOWN,
        track_status=TrackStatus.TENTATIVE, track_quality=0.3, last_update_time=1.0, sequence_number=1,
        reply_type=None, confidence=0.3, signal_strength=0.85, propagation_delay=1.0,
    )
    record = PipelineRunRecord(scenario=_scenario(), active_tracks=[track])
    rows = compute_track_statistics(record)
    assert len(rows) == 1
    row = rows[0]
    assert row.track_status == "Tentative"
    assert row.ever_confirmed is False
    assert row.duration is None
    assert row.replies_received is None
    assert row.average_range_m == pytest.approx(175.0)
    assert row.average_signal_strength == pytest.approx(0.85)


def test_write_track_statistics_csv_deterministic_with_blank_optional_fields(tmp_path):
    track = IFFTrack(
        track_id=3, aircraft_id="T3", ownship_id="OWNSHIP", time=1.0, range_m=50.0, azimuth_deg=0.0,
        elevation_deg=0.0, relative_velocity=None, mode=IFFMode.MODE_S, reply_status=MeasurementStatus.VALID,
        mode_s_address="A00003", authentication_result=False, friend_foe_status=FriendFoeStatus.UNKNOWN,
        track_status=TrackStatus.CONFIRMED, track_quality=1.0, last_update_time=1.0, sequence_number=1,
        reply_type=None, confidence=1.0, signal_strength=0.95, propagation_delay=1.0,
    )
    record = PipelineRunRecord(scenario=_scenario(), active_tracks=[track])
    rows = compute_track_statistics(record)
    path_a = write_track_statistics_csv(rows, tmp_path / "a.csv")
    path_b = write_track_statistics_csv(rows, tmp_path / "b.csv")
    assert path_a.read_text(encoding="utf-8") == path_b.read_text(encoding="utf-8")

    import csv as csv_module
    with path_a.open(encoding="utf-8") as handle:
        row = next(csv_module.DictReader(handle))
    assert row["Duration"] == ""
    assert row["Replies_Received"] == ""
