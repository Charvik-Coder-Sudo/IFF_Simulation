"""Tests for Phase 10 latency_analysis.py.

Covers: exact mean/min/max/stdev on synthetic measurements with known
delays, the documented-zero Scheduler/Track-Update delay components,
the receiver_delay_us sanity-check formula, and deterministic CSV
output.
"""

from __future__ import annotations

import statistics as pystats

import pytest

from iff_simulator.domain import Aircraft, AircraftState, Scenario, Vector3
from iff_simulator.sensors.iff import IFFMode, MeasurementStatus, ReceiverTickResult
from iff_simulator.sensors.iff.measurement import DecodedIFFMeasurement
from iff_simulator.analysis import PipelineRunRecord, compute_latency_breakdown, receiver_delay_us, write_latency_statistics_csv

ZERO = Vector3(0.0, 0.0, 0.0)


def _scenario():
    aircraft = [Aircraft(aircraft_id="OWNSHIP"), Aircraft(aircraft_id="T1")]
    history = {a.aircraft_id: [AircraftState(time=1.0, position=ZERO, velocity=ZERO)] for a in aircraft}
    return Scenario(aircraft, history)


def _measurement(time, processing_delay, propagation_delay, arrival_time=None, status=MeasurementStatus.VALID):
    if arrival_time is None:
        arrival_time = time + (processing_delay + propagation_delay) / 1_000_000.0
    return DecodedIFFMeasurement(
        measurement_id=1, time=time, target_id="T1", ownship_id="OWNSHIP", mode=IFFMode.MODE_S, range_m=100.0,
        azimuth_deg=0.0, elevation_deg=0.0, icao_address="A00001", authentication_result=False, identity="BLUE",
        mission=None, reply_status=status, processing_delay=processing_delay, propagation_delay=propagation_delay,
        arrival_time=arrival_time, sequence_number=1,
    )


def _tick(measurement):
    return ReceiverTickResult(real_measurement=measurement, false_alarm_measurements=[], fruited_measurements=[])


def test_processing_and_propagation_delay_stats_exact():
    measurements = [_measurement(time=1.0, processing_delay=p, propagation_delay=1.0) for p in (40.0, 50.0, 60.0)]
    record = PipelineRunRecord(scenario=_scenario(), tick_results=[_tick(m) for m in measurements])
    breakdown = compute_latency_breakdown(record)

    assert breakdown.processing_delay.mean == pytest.approx(50.0)
    assert breakdown.processing_delay.minimum == pytest.approx(40.0)
    assert breakdown.processing_delay.maximum == pytest.approx(60.0)
    assert breakdown.processing_delay.stdev == pytest.approx(pystats.pstdev([40.0, 50.0, 60.0]))


def test_total_end_to_end_delay_matches_arrival_minus_time():
    m = _measurement(time=2.0, processing_delay=50.0, propagation_delay=5.0)
    record = PipelineRunRecord(scenario=_scenario(), tick_results=[_tick(m)])
    breakdown = compute_latency_breakdown(record)
    expected_us = (m.arrival_time - m.time) * 1_000_000.0
    assert breakdown.total_end_to_end_delay.mean == pytest.approx(expected_us)
    assert breakdown.total_end_to_end_delay.mean == pytest.approx(55.0)


def test_scheduler_and_track_update_delay_are_documented_zero():
    measurements = [_measurement(time=1.0, processing_delay=50.0, propagation_delay=1.0) for _ in range(3)]
    record = PipelineRunRecord(scenario=_scenario(), tick_results=[_tick(m) for m in measurements])
    breakdown = compute_latency_breakdown(record)
    assert breakdown.scheduler_delay.mean == 0.0
    assert breakdown.scheduler_delay.stdev == 0.0
    assert breakdown.track_update_delay.mean == 0.0


def test_receiver_delay_is_zero_when_arrival_time_matches_construction():
    m = _measurement(time=1.0, processing_delay=50.0, propagation_delay=10.0)
    assert receiver_delay_us(m) == pytest.approx(0.0, abs=1e-9)


def test_receiver_delay_detects_a_nonzero_discrepancy():
    """If arrival_time doesn't match time+processing+propagation exactly
    (e.g. a hypothetical future receiver-internal delay), receiver_delay_us
    must surface it rather than silently reporting 0."""
    m = _measurement(time=1.0, processing_delay=50.0, propagation_delay=10.0, arrival_time=1.0001)
    delay = receiver_delay_us(m)
    assert delay == pytest.approx((1.0001 - (1.0 + 60.0 / 1_000_000.0)) * 1_000_000.0)
    assert delay != 0.0


def test_receiver_delay_none_when_timestamps_missing():
    no_reply = DecodedIFFMeasurement(
        measurement_id=1, time=1.0, target_id="T1", ownship_id="OWNSHIP", mode=IFFMode.MODE_S, range_m=0.0,
        azimuth_deg=0.0, elevation_deg=0.0, icao_address=None, authentication_result=False, identity="UNKNOWN",
        mission=None, reply_status=MeasurementStatus.NO_REPLY, processing_delay=None, propagation_delay=None,
        arrival_time=None, sequence_number=1,
    )
    assert receiver_delay_us(no_reply) is None


def test_empty_record_gives_zero_stats_everywhere():
    record = PipelineRunRecord(scenario=_scenario())
    breakdown = compute_latency_breakdown(record)
    for component in (
        breakdown.scheduler_delay, breakdown.processing_delay, breakdown.propagation_delay,
        breakdown.receiver_delay, breakdown.track_update_delay, breakdown.total_end_to_end_delay,
    ):
        assert component.mean == 0.0
        assert component.minimum == 0.0
        assert component.maximum == 0.0
        assert component.stdev == 0.0


def test_write_latency_statistics_csv_deterministic_and_has_six_rows(tmp_path):
    measurements = [_measurement(time=1.0, processing_delay=50.0, propagation_delay=1.0)]
    record = PipelineRunRecord(scenario=_scenario(), tick_results=[_tick(m) for m in measurements])
    breakdown = compute_latency_breakdown(record)

    path_a = write_latency_statistics_csv(breakdown, tmp_path / "a.csv")
    path_b = write_latency_statistics_csv(breakdown, tmp_path / "b.csv")
    assert path_a.read_text(encoding="utf-8") == path_b.read_text(encoding="utf-8")

    import csv as csv_module
    with path_a.open(encoding="utf-8") as handle:
        rows = list(csv_module.DictReader(handle))
    assert len(rows) == 6
    components = {row["Component"] for row in rows}
    assert components == {
        "Scheduler_Delay_Us", "Processing_Delay_Us", "Propagation_Delay_Us",
        "Receiver_Delay_Us", "Track_Update_Delay_Us", "Total_End_To_End_Delay_Us",
    }
