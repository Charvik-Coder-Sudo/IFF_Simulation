"""Confusion matrices: Ground Truth Identity vs Reported Identity, and
Ground Truth Authentication vs Reported Authentication (Phase 10).

Purpose:
    Implements a generic `ConfusionMatrix` (counts + per-class
    precision/recall/F1 + overall accuracy, from any list of
    `(true_label, predicted_label)` pairs), plus the two concrete
    matrices this phase asks for:
    `compute_identity_confusion_matrix` (3x3: FRIENDLY/FOE/UNKNOWN) and
    `compute_authentication_confusion_matrix` (2x2: AUTHENTICATED/FAILED,
    Mode 5 replies only).

Inputs:
    A `PipelineRunRecord`.

Outputs:
    Two `ConfusionMatrix` instances, and `write_confusion_matrix_csv`
    for `confusion_matrix.csv` (both matrices in one file, a `Matrix`
    column distinguishing them).

Engineering explanation:
    Ground Truth identity (`Aircraft.identity`, a legacy string like
    "FRIEND"/"FOE"/"NEUTRAL") and the pipeline's reported identity
    (`DecodedIFFMeasurement.identity`, "BLUE"/"RED"/"NEUTRAL"/"UNKNOWN")
    use two different vocabularies (see `authentication.py`'s own
    `_LEGACY_IDENTITY_TO_STATUS`). Rather than touching that existing
    mapping, this module defines its own local `_to_identity_class`
    folding *both* vocabularies into the exact three categories this
    phase's spec names (FRIENDLY/FOE/UNKNOWN -- NEUTRAL folds into
    UNKNOWN, since the spec lists only three buckets). The
    authentication ground-truth label is obtained by calling the
    existing, already-tested `AuthenticationEngine.authenticate`
    directly on `Scenario.get_aircraft(target_id)` -- never
    reimplementing that logic. False-alarm and fruited measurements are
    excluded from both matrices: they have no Ground Truth aircraft to
    compare against (see `false_replies.py`/`fruiting.py` -- their
    `target_id`s are fabricated, not real Scenario aircraft).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from ..sensors.iff import AuthenticationEngine, AuthenticationResult, IFFMode, MeasurementStatus
from .run_record import PipelineRunRecord
from .statistics import safe_divide

CONFUSION_MATRIX_CSV_COLUMNS = [
    "Matrix", "Row_Type", "True_Label", "Predicted_Label", "Count", "Precision", "Recall", "F1", "Accuracy",
]

IDENTITY_LABELS = ["FRIENDLY", "FOE", "UNKNOWN"]
AUTHENTICATION_LABELS = ["AUTHENTICATED", "FAILED"]

_FRIENDLY_ALIASES = {"FRIEND", "FRIENDLY", "BLUE"}
_FOE_ALIASES = {"FOE", "HOSTILE", "ENEMY", "RED"}


def _to_identity_class(identity: str) -> str:
    """Fold either Ground Truth (`Aircraft.identity`) or reported
    (`DecodedIFFMeasurement.identity`) vocabulary into exactly
    FRIENDLY/FOE/UNKNOWN. Anything not recognized as friendly or foe
    (including NEUTRAL/CIVIL) is UNKNOWN."""
    value = identity.upper()
    if value in _FRIENDLY_ALIASES:
        return "FRIENDLY"
    if value in _FOE_ALIASES:
        return "FOE"
    return "UNKNOWN"


@dataclass(frozen=True, slots=True)
class ConfusionMatrix:
    """A generic confusion matrix over any fixed label set.

    Purpose:
        Carry raw counts plus derived precision/recall/F1 (per class)
        and overall accuracy, computed once from a list of
        `(true_label, predicted_label)` pairs.
    """

    labels: list
    counts: dict  # counts[true_label][predicted_label] -> int
    precision: dict  # label -> float
    recall: dict  # label -> float
    f1: dict  # label -> float
    accuracy: float


def compute_confusion_matrix(pairs: list, labels: list) -> ConfusionMatrix:
    """Build a `ConfusionMatrix` from `(true_label, predicted_label)` pairs.

    Mathematics (per label L):
        TP = counts[L][L]
        FP = sum(counts[other][L] for other != L)
        FN = sum(counts[L][other] for other != L)
        precision(L) = TP / (TP + FP)   (0.0 if TP+FP == 0)
        recall(L)    = TP / (TP + FN)   (0.0 if TP+FN == 0)
        F1(L)        = 2 * precision * recall / (precision + recall)   (0.0 if both 0)
        accuracy     = sum(counts[L][L] for all L) / total pairs
    """
    counts = {true_label: {pred_label: 0 for pred_label in labels} for true_label in labels}
    for true_label, predicted_label in pairs:
        counts[true_label][predicted_label] += 1

    precision, recall, f1 = {}, {}, {}
    for label in labels:
        tp = counts[label][label]
        fp = sum(counts[other][label] for other in labels if other != label)
        fn = sum(counts[label][other] for other in labels if other != label)
        p = safe_divide(tp, tp + fp)
        r = safe_divide(tp, tp + fn)
        precision[label] = p
        recall[label] = r
        f1[label] = safe_divide(2 * p * r, p + r)

    total = len(pairs)
    correct = sum(counts[label][label] for label in labels)
    accuracy = safe_divide(correct, total)

    return ConfusionMatrix(labels=list(labels), counts=counts, precision=precision, recall=recall, f1=f1, accuracy=accuracy)


def compute_identity_confusion_matrix(record: PipelineRunRecord) -> ConfusionMatrix:
    """Ground Truth Identity vs Reported Identity, FRIENDLY/FOE/UNKNOWN."""
    pairs = []
    for tick in record.tick_results:
        measurement = tick.real_measurement
        if measurement is None or measurement.reply_status != MeasurementStatus.VALID:
            continue
        aircraft = record.scenario.get_aircraft(measurement.target_id)
        true_label = _to_identity_class(aircraft.identity)
        predicted_label = _to_identity_class(measurement.identity)
        pairs.append((true_label, predicted_label))
    return compute_confusion_matrix(pairs, IDENTITY_LABELS)


def compute_authentication_confusion_matrix(record: PipelineRunRecord) -> ConfusionMatrix:
    """Ground Truth Authentication vs Reported Authentication, Mode 5 only."""
    engine = AuthenticationEngine()
    pairs = []
    for tick in record.tick_results:
        measurement = tick.real_measurement
        if measurement is None or measurement.reply_status != MeasurementStatus.VALID:
            continue
        if measurement.mode not in (IFFMode.MODE5_L1, IFFMode.MODE5_L2):
            continue
        aircraft = record.scenario.get_aircraft(measurement.target_id)
        true_label = "AUTHENTICATED" if engine.authenticate(aircraft) else "FAILED"
        predicted_label = (
            "AUTHENTICATED" if measurement.authentication_status == AuthenticationResult.AUTHENTICATED else "FAILED"
        )
        pairs.append((true_label, predicted_label))
    return compute_confusion_matrix(pairs, AUTHENTICATION_LABELS)


def _matrix_to_rows(matrix_name: str, matrix: ConfusionMatrix) -> list:
    rows = []
    for true_label in matrix.labels:
        for predicted_label in matrix.labels:
            rows.append({
                "Matrix": matrix_name, "Row_Type": "CELL", "True_Label": true_label,
                "Predicted_Label": predicted_label, "Count": matrix.counts[true_label][predicted_label],
                "Precision": "", "Recall": "", "F1": "", "Accuracy": "",
            })
    for label in matrix.labels:
        rows.append({
            "Matrix": matrix_name, "Row_Type": "METRIC", "True_Label": label, "Predicted_Label": "",
            "Count": "", "Precision": matrix.precision[label], "Recall": matrix.recall[label],
            "F1": matrix.f1[label], "Accuracy": "",
        })
    rows.append({
        "Matrix": matrix_name, "Row_Type": "METRIC", "True_Label": "OVERALL", "Predicted_Label": "",
        "Count": "", "Precision": "", "Recall": "", "F1": "", "Accuracy": matrix.accuracy,
    })
    return rows


def write_confusion_matrix_csv(
    identity_matrix: ConfusionMatrix, authentication_matrix: ConfusionMatrix, output_path: Path | str
) -> Path:
    """Write both confusion matrices to `confusion_matrix.csv`."""
    rows = _matrix_to_rows("Identity", identity_matrix) + _matrix_to_rows("Authentication", authentication_matrix)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONFUSION_MATRIX_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_path
