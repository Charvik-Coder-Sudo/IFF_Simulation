"""Tests for Phase 10 confusion_matrix.py.

Covers: generic ConfusionMatrix correctness (precision/recall/F1/
accuracy against hand-computed expected values), the identity matrix's
FRIENDLY/FOE/UNKNOWN vocabulary mapping (from both Ground Truth's
legacy identity strings and the pipeline's reported BLUE/RED/NEUTRAL/
UNKNOWN strings), the authentication matrix restricted to Mode 5 only,
exclusion of false-alarm/fruited measurements from both matrices, and
deterministic CSV output.
"""

from __future__ import annotations

import pytest

from iff_simulator.domain import Aircraft, AircraftState, Scenario, Vector3
from iff_simulator.sensors.iff import IFFMode, MeasurementStatus, ReceiverTickResult
from iff_simulator.sensors.iff.authentication import AuthenticationResult
from iff_simulator.sensors.iff.measurement import DecodedIFFMeasurement
from iff_simulator.analysis import (
    AUTHENTICATION_LABELS,
    IDENTITY_LABELS,
    PipelineRunRecord,
    compute_authentication_confusion_matrix,
    compute_confusion_matrix,
    compute_identity_confusion_matrix,
    write_confusion_matrix_csv,
)

ZERO = Vector3(0.0, 0.0, 0.0)

_AUTH_MODE_DATA_TRUE = {"authentication_status": "AUTHENTICATED", "mode5_enabled": True, "crypto_key_present": True}
_AUTH_MODE_DATA_FALSE = {}  # deterministically fails AuthenticationEngine.authenticate


def _scenario(aircraft_list):
    history = {a.aircraft_id: [AircraftState(time=1.0, position=ZERO, velocity=ZERO)] for a in aircraft_list}
    return Scenario(aircraft_list, history)


def _measurement(
    target_id, mode=IFFMode.MODE_S, identity="BLUE", auth_status=AuthenticationResult.NOT_APPLICABLE,
    status=MeasurementStatus.VALID,
):
    return DecodedIFFMeasurement(
        measurement_id=1, time=1.0, target_id=target_id, ownship_id="OWNSHIP", mode=mode, range_m=100.0,
        azimuth_deg=0.0, elevation_deg=0.0, icao_address="A00001", authentication_result=False, identity=identity,
        mission=None, reply_status=status, processing_delay=50.0, propagation_delay=1.0, arrival_time=1.0,
        sequence_number=1, authentication_status=auth_status,
    )


def _tick(real=None, false_alarms=None, fruited=None):
    return ReceiverTickResult(real_measurement=real, false_alarm_measurements=false_alarms or [], fruited_measurements=fruited or [])


# ---------------------------------------------------------------------------
# Generic ConfusionMatrix
# ---------------------------------------------------------------------------


def test_generic_confusion_matrix_counts_and_metrics():
    # 3 true "A" (2 correctly predicted "A", 1 predicted "B")
    # 2 true "B" (both correctly predicted "B")
    pairs = [("A", "A"), ("A", "A"), ("A", "B"), ("B", "B"), ("B", "B")]
    matrix = compute_confusion_matrix(pairs, ["A", "B"])

    assert matrix.counts["A"]["A"] == 2
    assert matrix.counts["A"]["B"] == 1
    assert matrix.counts["B"]["B"] == 2
    assert matrix.counts["B"]["A"] == 0

    # Precision(A) = TP/(TP+FP) = 2/(2+0) = 1.0 ; Recall(A) = TP/(TP+FN) = 2/(2+1) = 0.666...
    assert matrix.precision["A"] == pytest.approx(1.0)
    assert matrix.recall["A"] == pytest.approx(2 / 3)
    assert matrix.f1["A"] == pytest.approx(2 * 1.0 * (2 / 3) / (1.0 + 2 / 3))

    # Precision(B) = TP/(TP+FP) = 2/(2+1) = 0.666... ; Recall(B) = TP/(TP+FN) = 2/2 = 1.0
    assert matrix.precision["B"] == pytest.approx(2 / 3)
    assert matrix.recall["B"] == pytest.approx(1.0)

    # Accuracy = correct / total = 4/5
    assert matrix.accuracy == pytest.approx(0.8)


def test_generic_confusion_matrix_empty_pairs_gives_zero_not_crash():
    matrix = compute_confusion_matrix([], ["A", "B"])
    assert matrix.accuracy == 0.0
    assert matrix.precision["A"] == 0.0
    assert matrix.recall["A"] == 0.0
    assert matrix.f1["A"] == 0.0


def test_generic_confusion_matrix_perfect_predictions_give_accuracy_one():
    pairs = [("A", "A")] * 5 + [("B", "B")] * 5
    matrix = compute_confusion_matrix(pairs, ["A", "B"])
    assert matrix.accuracy == 1.0
    assert matrix.precision["A"] == 1.0
    assert matrix.recall["A"] == 1.0
    assert matrix.f1["A"] == 1.0


# ---------------------------------------------------------------------------
# Identity confusion matrix
# ---------------------------------------------------------------------------


def test_identity_matrix_maps_ground_truth_and_reported_vocabularies_correctly():
    aircraft = [
        Aircraft(aircraft_id="OWNSHIP"),
        Aircraft(aircraft_id="T1", identity="FRIEND"),  # ground truth -> FRIENDLY
        Aircraft(aircraft_id="T2", identity="FOE"),  # ground truth -> FOE
        Aircraft(aircraft_id="T3", identity="NEUTRAL"),  # ground truth -> UNKNOWN (folds in)
    ]
    scenario = _scenario(aircraft)
    tick_results = [
        _tick(real=_measurement("T1", identity="BLUE")),  # reported -> FRIENDLY (correct)
        _tick(real=_measurement("T2", identity="RED")),  # reported -> FOE (correct)
        _tick(real=_measurement("T3", identity="NEUTRAL")),  # reported -> UNKNOWN (correct)
    ]
    record = PipelineRunRecord(scenario=scenario, tick_results=tick_results)
    matrix = compute_identity_confusion_matrix(record)
    assert matrix.labels == IDENTITY_LABELS
    assert matrix.counts["FRIENDLY"]["FRIENDLY"] == 1
    assert matrix.counts["FOE"]["FOE"] == 1
    assert matrix.counts["UNKNOWN"]["UNKNOWN"] == 1
    assert matrix.accuracy == 1.0


def test_identity_matrix_misclassification_reduces_accuracy():
    aircraft = [Aircraft(aircraft_id="OWNSHIP"), Aircraft(aircraft_id="T1", identity="FOE")]
    scenario = _scenario(aircraft)
    # Ground truth is FOE, but the pipeline reported BLUE (FRIENDLY) -- a misclassification.
    tick_results = [_tick(real=_measurement("T1", identity="BLUE"))]
    record = PipelineRunRecord(scenario=scenario, tick_results=tick_results)
    matrix = compute_identity_confusion_matrix(record)
    assert matrix.counts["FOE"]["FRIENDLY"] == 1
    assert matrix.accuracy == 0.0


def test_identity_matrix_excludes_no_reply_and_garbled_measurements():
    aircraft = [Aircraft(aircraft_id="OWNSHIP"), Aircraft(aircraft_id="T1", identity="FRIEND")]
    scenario = _scenario(aircraft)
    tick_results = [
        _tick(real=_measurement("T1", identity="UNKNOWN", status=MeasurementStatus.NO_REPLY)),
        _tick(real=_measurement("T1", identity="UNKNOWN", status=MeasurementStatus.GARBLED)),
    ]
    record = PipelineRunRecord(scenario=scenario, tick_results=tick_results)
    matrix = compute_identity_confusion_matrix(record)
    assert sum(sum(row.values()) for row in matrix.counts.values()) == 0


def test_identity_matrix_excludes_false_alarm_and_fruited_measurements():
    """False/fruited measurements have phantom target_ids with no Ground
    Truth aircraft -- they must never be included (and must not raise a
    KeyError from Scenario.get_aircraft)."""
    aircraft = [Aircraft(aircraft_id="OWNSHIP"), Aircraft(aircraft_id="T1", identity="FRIEND")]
    scenario = _scenario(aircraft)
    false_measurement = _measurement("FALSE-1", identity="UNKNOWN")
    fruited_measurement = _measurement("FRUIT-1", identity="UNKNOWN", status=MeasurementStatus.FRUITED)
    tick_results = [_tick(real=_measurement("T1", identity="BLUE"), false_alarms=[false_measurement], fruited=[fruited_measurement])]
    record = PipelineRunRecord(scenario=scenario, tick_results=tick_results)
    matrix = compute_identity_confusion_matrix(record)
    # Only T1's real measurement counted; FALSE-1/FRUIT-1 excluded (no Ground Truth aircraft for them)
    assert sum(sum(row.values()) for row in matrix.counts.values()) == 1


# ---------------------------------------------------------------------------
# Authentication confusion matrix
# ---------------------------------------------------------------------------


def test_authentication_matrix_ground_truth_via_authentication_engine():
    aircraft = [
        Aircraft(aircraft_id="OWNSHIP"),
        Aircraft(aircraft_id="T1", mode_data=_AUTH_MODE_DATA_TRUE),  # should authenticate
        Aircraft(aircraft_id="T2", mode_data=_AUTH_MODE_DATA_FALSE),  # should not authenticate
    ]
    scenario = _scenario(aircraft)
    tick_results = [
        _tick(real=_measurement("T1", mode=IFFMode.MODE5_L1, auth_status=AuthenticationResult.AUTHENTICATED)),
        _tick(real=_measurement("T2", mode=IFFMode.MODE5_L1, auth_status=AuthenticationResult.FAILED)),
    ]
    record = PipelineRunRecord(scenario=scenario, tick_results=tick_results)
    matrix = compute_authentication_confusion_matrix(record)
    assert matrix.labels == AUTHENTICATION_LABELS
    assert matrix.counts["AUTHENTICATED"]["AUTHENTICATED"] == 1
    assert matrix.counts["FAILED"]["FAILED"] == 1
    assert matrix.accuracy == 1.0


def test_authentication_matrix_excludes_mode_s():
    aircraft = [Aircraft(aircraft_id="OWNSHIP"), Aircraft(aircraft_id="T1", mode_data=_AUTH_MODE_DATA_TRUE)]
    scenario = _scenario(aircraft)
    tick_results = [_tick(real=_measurement("T1", mode=IFFMode.MODE_S))]
    record = PipelineRunRecord(scenario=scenario, tick_results=tick_results)
    matrix = compute_authentication_confusion_matrix(record)
    assert sum(sum(row.values()) for row in matrix.counts.values()) == 0


def test_authentication_matrix_mismatch_reduces_accuracy():
    aircraft = [Aircraft(aircraft_id="OWNSHIP"), Aircraft(aircraft_id="T1", mode_data=_AUTH_MODE_DATA_TRUE)]
    scenario = _scenario(aircraft)
    # Ground truth says T1 should authenticate, but the reported status is FAILED.
    tick_results = [_tick(real=_measurement("T1", mode=IFFMode.MODE5_L2, auth_status=AuthenticationResult.FAILED))]
    record = PipelineRunRecord(scenario=scenario, tick_results=tick_results)
    matrix = compute_authentication_confusion_matrix(record)
    assert matrix.counts["AUTHENTICATED"]["FAILED"] == 1
    assert matrix.accuracy == 0.0


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------


def test_write_confusion_matrix_csv_deterministic(tmp_path):
    aircraft = [Aircraft(aircraft_id="OWNSHIP"), Aircraft(aircraft_id="T1", identity="FRIEND", mode_data=_AUTH_MODE_DATA_TRUE)]
    scenario = _scenario(aircraft)
    tick_results = [_tick(real=_measurement("T1", mode=IFFMode.MODE5_L1, identity="BLUE", auth_status=AuthenticationResult.AUTHENTICATED))]
    record = PipelineRunRecord(scenario=scenario, tick_results=tick_results)
    identity_matrix = compute_identity_confusion_matrix(record)
    auth_matrix = compute_authentication_confusion_matrix(record)

    path_a = write_confusion_matrix_csv(identity_matrix, auth_matrix, tmp_path / "a.csv")
    path_b = write_confusion_matrix_csv(identity_matrix, auth_matrix, tmp_path / "b.csv")
    assert path_a.read_text(encoding="utf-8") == path_b.read_text(encoding="utf-8")

    content = path_a.read_text(encoding="utf-8")
    assert "Identity" in content
    assert "Authentication" in content
