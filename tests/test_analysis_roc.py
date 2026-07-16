"""Tests for Phase 10 roc_analysis.py.

Covers: monotonic FPR/TPR sweep behavior, AUC bounds [0, 1], perfect
separability gives AUC~=1, no separability gives AUC~=0.5, and the
"only replies that reached propagation" universe (Pd-rejected replies
with no signal_strength are correctly excluded).
"""

from __future__ import annotations

import pytest

from iff_simulator.domain import Aircraft, AircraftState, Scenario, Vector3
from iff_simulator.sensors.iff import IFFMode, MeasurementStatus, ReceiverTickResult
from iff_simulator.sensors.iff.measurement import DecodedIFFMeasurement
from iff_simulator.analysis import PipelineRunRecord, compute_roc_curve

ZERO = Vector3(0.0, 0.0, 0.0)


def _scenario():
    aircraft = [Aircraft(aircraft_id="OWNSHIP"), Aircraft(aircraft_id="T1")]
    history = {a.aircraft_id: [AircraftState(time=1.0, position=ZERO, velocity=ZERO)] for a in aircraft}
    return Scenario(aircraft, history)


def _measurement(target_id, signal_strength, status=MeasurementStatus.VALID):
    return DecodedIFFMeasurement(
        measurement_id=1, time=1.0, target_id=target_id, ownship_id="OWNSHIP", mode=IFFMode.MODE_S,
        range_m=100.0, azimuth_deg=0.0, elevation_deg=0.0, icao_address="A00001", authentication_result=False,
        identity="BLUE", mission=None, reply_status=status, processing_delay=50.0, propagation_delay=1.0,
        arrival_time=1.0, sequence_number=1, signal_strength=signal_strength,
    )


def _tick(real=None, false_alarms=None, fruited=None):
    return ReceiverTickResult(real_measurement=real, false_alarm_measurements=false_alarms or [], fruited_measurements=fruited or [])


def test_no_samples_gives_zero_rates_everywhere_no_crash():
    record = PipelineRunRecord(scenario=_scenario())
    roc = compute_roc_curve(record, thresholds=[0.0, 0.5, 1.0])
    assert roc.true_positive_rate == [0.0, 0.0, 0.0]
    assert roc.false_positive_rate == [0.0, 0.0, 0.0]
    assert roc.area_under_curve == 0.0


def test_perfect_separability_gives_auc_near_one():
    """Real replies all have high signal strength; false/fruited replies
    all have low signal strength -- a threshold in between perfectly
    separates them."""
    tick_results = []
    for i in range(20):
        tick_results.append(_tick(real=_measurement("T1", signal_strength=0.9)))
    for i in range(20):
        tick_results.append(_tick(false_alarms=[_measurement(f"FALSE-{i}", signal_strength=0.1)]))
    record = PipelineRunRecord(scenario=_scenario(), tick_results=tick_results)
    roc = compute_roc_curve(record)
    assert roc.area_under_curve == pytest.approx(1.0, abs=0.05)


def test_no_separability_gives_auc_near_half():
    """Real and false/fruited replies share the exact same signal
    strength distribution -- no threshold can discriminate them."""
    strengths = [0.1, 0.3, 0.5, 0.7, 0.9] * 10
    tick_results = []
    for i, s in enumerate(strengths):
        if i % 2 == 0:
            tick_results.append(_tick(real=_measurement("T1", signal_strength=s)))
        else:
            tick_results.append(_tick(false_alarms=[_measurement(f"FALSE-{i}", signal_strength=s)]))
    record = PipelineRunRecord(scenario=_scenario(), tick_results=tick_results)
    roc = compute_roc_curve(record)
    assert roc.area_under_curve == pytest.approx(0.5, abs=0.1)


def test_tpr_and_fpr_are_monotonically_non_increasing_as_threshold_rises():
    tick_results = [_tick(real=_measurement("T1", signal_strength=s)) for s in (0.2, 0.4, 0.6, 0.8)]
    tick_results += [_tick(false_alarms=[_measurement(f"FALSE-{i}", signal_strength=s)]) for i, s in enumerate((0.1, 0.3, 0.5, 0.7))]
    record = PipelineRunRecord(scenario=_scenario(), tick_results=tick_results)
    thresholds = [0.0, 0.25, 0.5, 0.75, 1.0]
    roc = compute_roc_curve(record, thresholds=thresholds)
    assert all(a >= b for a, b in zip(roc.true_positive_rate, roc.true_positive_rate[1:]))
    assert all(a >= b for a, b in zip(roc.false_positive_rate, roc.false_positive_rate[1:]))


def test_auc_always_within_unit_interval():
    tick_results = [_tick(real=_measurement("T1", signal_strength=0.5))]
    tick_results += [_tick(fruited=[_measurement("FRUIT-1", signal_strength=0.6)])]
    record = PipelineRunRecord(scenario=_scenario(), tick_results=tick_results)
    roc = compute_roc_curve(record)
    assert 0.0 <= roc.area_under_curve <= 1.0


def test_measurements_without_signal_strength_are_excluded_from_universe():
    """A NO_REPLY measurement (e.g. Pd-rejected before propagation) has
    signal_strength=None and must not distort the ROC universe."""
    no_signal = _measurement("T1", signal_strength=None, status=MeasurementStatus.NO_REPLY)
    with_signal = _measurement("T2", signal_strength=0.7)
    record = PipelineRunRecord(
        scenario=_scenario(), tick_results=[_tick(real=no_signal), _tick(real=with_signal)],
    )
    roc = compute_roc_curve(record, thresholds=[0.5])
    # only the one measurement with a known signal_strength counts
    assert roc.true_positive_rate == [1.0]


def test_default_thresholds_span_observed_signal_strength_range():
    tick_results = [_tick(real=_measurement("T1", signal_strength=0.3)), _tick(real=_measurement("T2", signal_strength=0.8))]
    record = PipelineRunRecord(scenario=_scenario(), tick_results=tick_results)
    roc = compute_roc_curve(record)
    assert min(roc.thresholds) == pytest.approx(0.3)
    assert max(roc.thresholds) == pytest.approx(0.8)


def test_roc_curve_to_csv_rows_shape():
    record = PipelineRunRecord(scenario=_scenario(), tick_results=[_tick(real=_measurement("T1", signal_strength=0.5))])
    roc = compute_roc_curve(record, thresholds=[0.0, 0.5, 1.0])
    rows = roc.to_csv_rows()
    assert len(rows) == 3
    assert set(rows[0].keys()) == {"Threshold", "True_Positive_Rate", "False_Positive_Rate"}
