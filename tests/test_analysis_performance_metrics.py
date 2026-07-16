"""Tests for Phase 10 performance_metrics.py.

Covers: perfect detection, missed detections, false replies,
authentication failures, garbled replies (decoder success rate), track
confirmation/lifetime, detection range, delay averages, SNR proxy, and
zero-division safety on an empty record.
"""

from __future__ import annotations

import math

import pytest

from iff_simulator.domain import Aircraft, AircraftState, Scenario, Vector3
from iff_simulator.sensors.iff import (
    AuthenticationResult,
    IFFMode,
    InterrogationMessage,
    MeasurementStatus,
    ReceiverStatistics,
    ReceiverTickResult,
    TrackStatus,
    UplinkFormat,
)
from iff_simulator.sensors.iff.measurement import DecodedIFFMeasurement
from iff_simulator.sensors.iff.track import IFFTrack, FriendFoeStatus
from iff_simulator.sensors.iff.track_summary import TrackSummary
from iff_simulator.analysis import NOISE_FLOOR, PipelineRunRecord, compute_performance_metrics

ZERO = Vector3(0.0, 0.0, 0.0)


def _scenario(aircraft_list=None):
    aircraft_list = aircraft_list or [Aircraft(aircraft_id="OWNSHIP"), Aircraft(aircraft_id="T1")]
    history = {a.aircraft_id: [AircraftState(time=1.0, position=ZERO, velocity=ZERO)] for a in aircraft_list}
    return Scenario(aircraft_list, history)


def _interrogation(seq, target_id="T1", range_m=100.0, mode=IFFMode.MODE_S, time=1.0):
    return InterrogationMessage(
        time=time, sequence_number=seq, ownship_id="OWNSHIP", target_id=target_id,
        mode=mode, uplink_format=UplinkFormat.UF11, range_m=range_m, azimuth_deg=0.0, elevation_deg=0.0,
    )


def _valid_measurement(
    seq=1, target_id="T1", time=1.0, range_m=100.0, mode=IFFMode.MODE_S, identity="BLUE",
    auth_result=False, auth_status=AuthenticationResult.NOT_APPLICABLE, signal_strength=0.9,
    processing_delay=50.0, propagation_delay=1.0, arrival_time=None,
):
    if arrival_time is None:
        arrival_time = time + (processing_delay + propagation_delay) / 1_000_000.0
    return DecodedIFFMeasurement(
        measurement_id=seq, time=time, target_id=target_id, ownship_id="OWNSHIP", mode=mode,
        range_m=range_m, azimuth_deg=0.0, elevation_deg=0.0,
        icao_address="A00001" if mode == IFFMode.MODE_S else None,
        authentication_result=auth_result, identity=identity, mission=None,
        reply_status=MeasurementStatus.VALID, processing_delay=processing_delay,
        propagation_delay=propagation_delay, arrival_time=arrival_time, sequence_number=seq,
        authentication_status=auth_status, signal_strength=signal_strength,
    )


def _no_reply_measurement(seq=1, target_id="T1", time=1.0):
    return DecodedIFFMeasurement(
        measurement_id=seq, time=time, target_id=target_id, ownship_id="OWNSHIP", mode=IFFMode.MODE_S,
        range_m=0.0, azimuth_deg=0.0, elevation_deg=0.0, icao_address=None, authentication_result=False,
        identity="UNKNOWN", mission=None, reply_status=MeasurementStatus.NO_REPLY, processing_delay=None,
        propagation_delay=None, arrival_time=None, sequence_number=seq,
    )


def _garbled_measurement(seq=1, target_id="T1", time=1.0):
    return DecodedIFFMeasurement(
        measurement_id=seq, time=time, target_id=target_id, ownship_id="OWNSHIP", mode=IFFMode.MODE_S,
        range_m=0.0, azimuth_deg=0.0, elevation_deg=0.0, icao_address=None, authentication_result=False,
        identity="UNKNOWN", mission=None, reply_status=MeasurementStatus.GARBLED, processing_delay=None,
        propagation_delay=None, arrival_time=None, sequence_number=seq,
    )


def _tick(real=None, false_alarms=None, fruited=None):
    return ReceiverTickResult(
        real_measurement=real, false_alarm_measurements=false_alarms or [], fruited_measurements=fruited or [],
    )


def _dummy_reply(seq=1):
    """A minimal stand-in `ReplyMessage`-shaped truthy value -- performance_metrics
    only ever checks `reply is not None`, never any of its fields."""
    return object()


def test_perfect_detection_gives_pd_one_and_full_success_rates():
    interrogations = [_interrogation(seq=n) for n in range(1, 6)]
    replies = [_dummy_reply(n) for n in range(1, 6)]
    tick_results = [_tick(real=_valid_measurement(seq=n)) for n in range(1, 6)]
    stats = ReceiverStatistics(
        replies_received=5, replies_lost=0, replies_garbled=0, replies_fruited=0, false_replies=0,
        average_detection_probability=1.0, average_signal_strength=0.9, average_delay_us=51.0, receiver_load=1.0,
    )
    record = PipelineRunRecord(
        scenario=_scenario(), interrogations=interrogations, replies=replies, tick_results=tick_results,
        receiver_statistics=stats,
    )
    metrics = compute_performance_metrics(record)
    assert metrics.detection_probability == 1.0
    assert metrics.reply_success_rate == 1.0
    assert metrics.decoder_success_rate == 1.0
    assert metrics.false_alarm_rate == 0.0


def test_missed_detections_reduce_detection_probability():
    interrogations = [_interrogation(seq=n) for n in range(1, 11)]
    replies = [_dummy_reply(n) for n in range(1, 11)]
    tick_results = [
        _tick(real=_valid_measurement(seq=n)) if n <= 7 else _tick(real=_no_reply_measurement(seq=n))
        for n in range(1, 11)
    ]
    record = PipelineRunRecord(
        scenario=_scenario(), interrogations=interrogations, replies=replies, tick_results=tick_results,
        receiver_statistics=ReceiverStatistics(7, 3, 0, 0, 0, 0.7, 0.9, 51.0, 1.0),
    )
    metrics = compute_performance_metrics(record)
    assert metrics.detection_probability == pytest.approx(0.7)
    assert metrics.reply_success_rate == pytest.approx(0.7)


def test_no_replies_at_all_gives_zero_not_crash():
    interrogations = [_interrogation(seq=n) for n in range(1, 4)]
    replies = [None, None, None]
    tick_results = [_tick(real=_no_reply_measurement(seq=n)) for n in range(1, 4)]
    record = PipelineRunRecord(
        scenario=_scenario(), interrogations=interrogations, replies=replies, tick_results=tick_results,
        receiver_statistics=ReceiverStatistics(0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0),
    )
    metrics = compute_performance_metrics(record)
    assert metrics.detection_probability == 0.0
    assert metrics.reply_success_rate == 0.0


def test_false_alarm_rate_matches_receiver_statistics_ratio():
    stats = ReceiverStatistics(
        replies_received=90, replies_lost=5, replies_garbled=0, replies_fruited=5, false_replies=5,
        average_detection_probability=0.9, average_signal_strength=0.9, average_delay_us=50.0, receiver_load=1.0,
    )
    record = PipelineRunRecord(scenario=_scenario(), receiver_statistics=stats)
    metrics = compute_performance_metrics(record)
    # false_replies / (received + false + fruited) == 5 / (90+5+5) == 0.05
    assert metrics.false_alarm_rate == pytest.approx(0.05)


def test_authentication_success_rate_over_mode5_replies_only():
    interrogations = [_interrogation(seq=n, mode=IFFMode.MODE5_L1) for n in range(1, 5)]
    replies = [_dummy_reply(n) for n in range(1, 5)]
    tick_results = [
        _tick(real=_valid_measurement(
            seq=n, mode=IFFMode.MODE5_L1,
            auth_status=AuthenticationResult.AUTHENTICATED if n <= 3 else AuthenticationResult.FAILED,
        ))
        for n in range(1, 5)
    ]
    record = PipelineRunRecord(
        scenario=_scenario(), interrogations=interrogations, replies=replies, tick_results=tick_results,
        receiver_statistics=ReceiverStatistics(4, 0, 0, 0, 0, 1.0, 0.9, 51.0, 1.0),
    )
    metrics = compute_performance_metrics(record)
    assert metrics.authentication_success_rate == pytest.approx(0.75)


def test_authentication_success_rate_zero_when_no_mode5_replies():
    interrogations = [_interrogation(seq=1, mode=IFFMode.MODE_S)]
    tick_results = [_tick(real=_valid_measurement(seq=1, mode=IFFMode.MODE_S))]
    record = PipelineRunRecord(
        scenario=_scenario(), interrogations=interrogations, replies=[_dummy_reply(1)], tick_results=tick_results,
        receiver_statistics=ReceiverStatistics(1, 0, 0, 0, 0, 1.0, 0.9, 51.0, 1.0),
    )
    metrics = compute_performance_metrics(record)
    assert metrics.authentication_success_rate == 0.0


def test_decoder_success_rate_accounts_for_garbled_replies():
    interrogations = [_interrogation(seq=n) for n in range(1, 5)]
    tick_results = [
        _tick(real=_valid_measurement(seq=n)) if n <= 3 else _tick(real=_garbled_measurement(seq=n))
        for n in range(1, 5)
    ]
    record = PipelineRunRecord(
        scenario=_scenario(), interrogations=interrogations, replies=[_dummy_reply(n) for n in range(1, 5)],
        tick_results=tick_results, receiver_statistics=ReceiverStatistics(3, 0, 1, 0, 0, 0.75, 0.9, 51.0, 1.0),
    )
    metrics = compute_performance_metrics(record)
    # decoded (3 VALID) / (3 VALID + 1 GARBLED) == 0.75
    assert metrics.decoder_success_rate == pytest.approx(0.75)


def _track_summary(track_id, aircraft_id, duration, confirmed_time):
    return TrackSummary(
        track_id=track_id, aircraft_id=aircraft_id, ownship_id="OWNSHIP", track_start_time=0.0,
        track_end_time=duration, duration=duration, replies_received=3, replies_missed=1,
        tentative_time=duration - confirmed_time, confirmed_time=confirmed_time, lost_time=0.0,
        max_range_m=200.0, min_range_m=100.0, avg_range_m=150.0, max_signal_strength=0.9,
        avg_signal_strength=0.8, final_track_status=TrackStatus.LOST,
    )


def _active_track(track_id, aircraft_id, status):
    return IFFTrack(
        track_id=track_id, aircraft_id=aircraft_id, ownship_id="OWNSHIP", time=1.0, range_m=150.0,
        azimuth_deg=0.0, elevation_deg=0.0, relative_velocity=None, mode=IFFMode.MODE_S,
        reply_status=MeasurementStatus.VALID, mode_s_address="A00001", authentication_result=False,
        friend_foe_status=FriendFoeStatus.UNKNOWN, track_status=status, track_quality=0.5, last_update_time=1.0,
        sequence_number=1, reply_type=None, confidence=0.5, signal_strength=0.9, propagation_delay=1.0,
    )


def test_track_confirmation_rate_over_completed_and_active_tracks():
    completed = [
        _track_summary(1, "T1", duration=10.0, confirmed_time=5.0),  # was confirmed
        _track_summary(2, "T2", duration=3.0, confirmed_time=0.0),  # never confirmed
    ]
    active = [
        _active_track(3, "T3", TrackStatus.CONFIRMED),
        _active_track(4, "T4", TrackStatus.TENTATIVE),
    ]
    record = PipelineRunRecord(scenario=_scenario(), completed_track_summaries=completed, active_tracks=active)
    metrics = compute_performance_metrics(record)
    # 2 of 4 tracks (T1, T3) ever confirmed
    assert metrics.track_confirmation_rate == pytest.approx(0.5)


def test_average_track_lifetime_over_completed_tracks_only():
    completed = [_track_summary(1, "T1", duration=10.0, confirmed_time=5.0), _track_summary(2, "T2", duration=20.0, confirmed_time=0.0)]
    record = PipelineRunRecord(scenario=_scenario(), completed_track_summaries=completed)
    metrics = compute_performance_metrics(record)
    assert metrics.average_track_lifetime_s == pytest.approx(15.0)


def test_detection_range_average_and_maximum():
    tick_results = [
        _tick(real=_valid_measurement(seq=1, range_m=100.0)),
        _tick(real=_valid_measurement(seq=2, range_m=300.0)),
    ]
    record = PipelineRunRecord(
        scenario=_scenario(),
        interrogations=[_interrogation(1), _interrogation(2)],
        replies=[_dummy_reply(1), _dummy_reply(2)],
        tick_results=tick_results,
        receiver_statistics=ReceiverStatistics(2, 0, 0, 0, 0, 1.0, 0.9, 51.0, 1.0),
    )
    metrics = compute_performance_metrics(record)
    assert metrics.average_detection_range_m == pytest.approx(200.0)
    assert metrics.maximum_detection_range_m == pytest.approx(300.0)


def test_processing_and_propagation_and_total_delay_averages():
    m = _valid_measurement(seq=1, processing_delay=50.0, propagation_delay=10.0, time=1.0)
    tick_results = [_tick(real=m)]
    record = PipelineRunRecord(
        scenario=_scenario(), interrogations=[_interrogation(1)], replies=[_dummy_reply(1)],
        tick_results=tick_results, receiver_statistics=ReceiverStatistics(1, 0, 0, 0, 0, 1.0, 0.9, 60.0, 1.0),
    )
    metrics = compute_performance_metrics(record)
    assert metrics.average_processing_delay_us == pytest.approx(50.0)
    assert metrics.average_propagation_delay_us == pytest.approx(10.0)
    # total delay == arrival_time - time, in microseconds == processing + propagation exactly here
    assert metrics.average_total_delay_us == pytest.approx(60.0)
    # receiver delay is ~0 by construction (arrival_time built from processing+propagation exactly)
    assert metrics.average_receiver_delay_us == pytest.approx(0.0, abs=1e-6)
    assert metrics.average_track_update_delay_s == 0.0


def test_average_snr_db_proxy_matches_formula():
    signal_strength = 0.5
    tick_results = [_tick(real=_valid_measurement(seq=1, signal_strength=signal_strength))]
    record = PipelineRunRecord(
        scenario=_scenario(), interrogations=[_interrogation(1)], replies=[_dummy_reply(1)],
        tick_results=tick_results, receiver_statistics=ReceiverStatistics(1, 0, 0, 0, 0, 1.0, signal_strength, 51.0, 1.0),
    )
    metrics = compute_performance_metrics(record)
    expected = 10.0 * math.log10(signal_strength / NOISE_FLOOR)
    assert metrics.average_snr_db_proxy == pytest.approx(expected)
    assert metrics.average_signal_strength == pytest.approx(signal_strength)


def test_empty_record_gives_all_zero_metrics_without_crashing():
    record = PipelineRunRecord(scenario=_scenario())
    metrics = compute_performance_metrics(record)
    assert metrics.detection_probability == 0.0
    assert metrics.false_alarm_rate == 0.0
    assert metrics.authentication_success_rate == 0.0
    assert metrics.reply_success_rate == 0.0
    assert metrics.decoder_success_rate == 0.0
    assert metrics.track_confirmation_rate == 0.0
    assert metrics.average_track_lifetime_s == 0.0
    assert metrics.average_detection_range_m == 0.0
    assert metrics.maximum_detection_range_m == 0.0
    assert metrics.average_signal_strength == 0.0
    assert metrics.average_snr_db_proxy == 0.0
