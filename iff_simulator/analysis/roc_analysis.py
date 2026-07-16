"""ROC analysis for the receiver's sensitivity threshold (Phase 10).

Purpose:
    Implements `compute_roc_curve`, which sweeps candidate
    `signal_strength` thresholds and reports the True Positive Rate,
    False Positive Rate, and Area Under Curve a receiver's
    `sensitivity_threshold` (Part 5 of Phase 9) would achieve at each
    one.

Inputs:
    A `PipelineRunRecord`, and an optional explicit list of thresholds
    (defaults to 50 evenly spaced points spanning the observed
    `signal_strength` range).

Outputs:
    A `RocCurve` (thresholds, true_positive_rate, false_positive_rate,
    area_under_curve).

Engineering explanation:
    This is deliberately scoped to what is actually measurable from
    existing pipeline output, not a general Pd-vs-Pfa curve:
    `ReceiverEffectsPipeline._submit_real` (Phase 9) only computes
    `signal_strength` for a real reply that already survived the Pd
    roll, so a reply lost to Pd never has a `signal_strength` value to
    sweep a threshold against. The universe here is therefore every
    measurement (real, false-alarm, or fruited) that *did* reach
    propagation -- i.e. has a non-`None` `signal_strength` -- asking
    "how well would a signal-strength threshold separate real
    transponder replies from false/fruited noise, among things the
    receiver's front end actually saw." See docs/METRICS.md for the
    full reasoning. This never modifies `ReceiverEffectsPipeline` or
    re-runs the simulation -- it is a pure post-hoc analysis of already-
    produced `DecodedIFFMeasurement`s.
"""

from __future__ import annotations

from dataclasses import dataclass

from .run_record import PipelineRunRecord
from .statistics import safe_divide

ROC_CURVE_CSV_COLUMNS = ["Threshold", "True_Positive_Rate", "False_Positive_Rate"]

DEFAULT_ROC_POINTS = 50


@dataclass(frozen=True, slots=True)
class RocCurve:
    """A swept ROC curve over `signal_strength` thresholds."""

    thresholds: list
    true_positive_rate: list
    false_positive_rate: list
    area_under_curve: float

    def to_csv_rows(self) -> list:
        return [
            {"Threshold": t, "True_Positive_Rate": tpr, "False_Positive_Rate": fpr}
            for t, tpr, fpr in zip(self.thresholds, self.true_positive_rate, self.false_positive_rate)
        ]


def _labeled_signal_strengths(record: PipelineRunRecord):
    """Yield `(signal_strength, is_real)` for every measurement (real,
    false-alarm, or fruited) this run that has a known signal strength."""
    for tick in record.tick_results:
        if tick.real_measurement is not None and tick.real_measurement.signal_strength is not None:
            yield tick.real_measurement.signal_strength, True
        for measurement in tick.false_alarm_measurements:
            if measurement.signal_strength is not None:
                yield measurement.signal_strength, False
        for measurement in tick.fruited_measurements:
            if measurement.signal_strength is not None:
                yield measurement.signal_strength, False


def compute_roc_curve(record: PipelineRunRecord, thresholds: list | None = None) -> RocCurve:
    """Sweep `thresholds` over observed `signal_strength` values and
    compute TPR/FPR/AUC.

    Outputs:
        A `RocCurve`. If there are no real samples (or no false/fruited
        samples), the corresponding rate is `0.0` at every threshold
        (via `safe_divide`) rather than raising or producing `NaN`.
    """
    samples = list(_labeled_signal_strengths(record))
    real_strengths = [s for s, is_real in samples if is_real]
    noise_strengths = [s for s, is_real in samples if not is_real]

    if thresholds is None:
        all_strengths = [s for s, _ in samples]
        if all_strengths:
            lo, hi = min(all_strengths), max(all_strengths)
        else:
            lo, hi = 0.0, 1.0
        step = (hi - lo) / (DEFAULT_ROC_POINTS - 1) if hi > lo else 0.0
        thresholds = [lo + step * i for i in range(DEFAULT_ROC_POINTS)] if step else [lo] * DEFAULT_ROC_POINTS

    tpr_values = [safe_divide(sum(1 for s in real_strengths if s >= t), len(real_strengths)) for t in thresholds]
    fpr_values = [safe_divide(sum(1 for s in noise_strengths if s >= t), len(noise_strengths)) for t in thresholds]

    auc = _trapezoidal_auc(fpr_values, tpr_values)

    return RocCurve(
        thresholds=list(thresholds), true_positive_rate=tpr_values, false_positive_rate=fpr_values,
        area_under_curve=auc,
    )


def _trapezoidal_auc(fpr_values: list, tpr_values: list) -> float:
    """Area under the (FPR, TPR) curve via the trapezoidal rule, with
    points sorted by ascending FPR first (thresholds are not
    necessarily already FPR-ordered when supplied explicitly)."""
    points = sorted(zip(fpr_values, tpr_values))
    if len(points) < 2:
        return 0.0
    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        area += (x1 - x0) * (y0 + y1) / 2.0
    return area
