"""Ties every Phase 10 analysis module together for one pipeline run.

Purpose:
    Implements `AnalysisReportGenerator`, the single entry point a
    caller (e.g. `run_analysis.py`) uses to compute every metric,
    write every required CSV, render every plot, and produce a
    generated per-run engineering-report markdown summary, from one
    `PipelineRunRecord`.

Inputs:
    A `PipelineRunRecord`.

Outputs:
    `.compute_all()` -> dict of every computed metrics object;
    `.write_csv_outputs(output_dir)` -> the 6 required CSVs;
    `.write_plots(output_dir, pd_model, pd_params, max_range_m)` -> the
    9 required PNGs; `.write_engineering_report(output_dir)` -> one
    generated markdown file summarizing this run's numbers (distinct
    from the static, hand-authored docs under `analysis/docs/`).

Engineering explanation:
    Purely a composition root -- every computation is delegated to the
    module that owns it (`performance_metrics.py`, `roc_analysis.py`,
    `confusion_matrix.py`, `latency_analysis.py`, `statistics.py`,
    `plots.py`); this file contains no metric formulas of its own.
"""

from __future__ import annotations

from pathlib import Path

from .confusion_matrix import compute_authentication_confusion_matrix, compute_identity_confusion_matrix, write_confusion_matrix_csv
from .latency_analysis import compute_latency_breakdown, write_latency_statistics_csv
from .performance_metrics import compute_performance_metrics, write_performance_metrics_csv
from .plots import AnalysisPlotter
from .roc_analysis import compute_roc_curve
from .run_record import PipelineRunRecord
from .statistics import (
    compute_authentication_statistics,
    compute_detection_statistics,
    compute_track_statistics,
    write_authentication_statistics_csv,
    write_detection_statistics_csv,
    write_track_statistics_csv,
)


class AnalysisReportGenerator:
    """Computes every Phase 10 metric and writes every artifact for one run.

    Inputs:
        record: the `PipelineRunRecord` to analyze.
    """

    def __init__(self, record: PipelineRunRecord) -> None:
        self.record = record

    def compute_all(self) -> dict:
        """Return every computed metrics object, keyed by name."""
        return {
            "performance_metrics": compute_performance_metrics(self.record),
            "detection_statistics": compute_detection_statistics(self.record),
            "authentication_statistics": compute_authentication_statistics(self.record),
            "track_statistics": compute_track_statistics(self.record),
            "roc_curve": compute_roc_curve(self.record),
            "identity_confusion_matrix": compute_identity_confusion_matrix(self.record),
            "authentication_confusion_matrix": compute_authentication_confusion_matrix(self.record),
            "latency_breakdown": compute_latency_breakdown(self.record),
        }

    def write_csv_outputs(self, output_dir: Path | str) -> dict:
        """Write all 6 required CSVs to `output_dir`, returning their paths by name."""
        results = self.compute_all()
        output_dir = Path(output_dir)
        return {
            "performance_metrics": write_performance_metrics_csv(
                results["performance_metrics"], output_dir / "performance_metrics.csv"
            ),
            "confusion_matrix": write_confusion_matrix_csv(
                results["identity_confusion_matrix"], results["authentication_confusion_matrix"],
                output_dir / "confusion_matrix.csv",
            ),
            "latency_statistics": write_latency_statistics_csv(
                results["latency_breakdown"], output_dir / "latency_statistics.csv"
            ),
            "detection_statistics": write_detection_statistics_csv(
                results["detection_statistics"], output_dir / "detection_statistics.csv"
            ),
            "authentication_statistics": write_authentication_statistics_csv(
                results["authentication_statistics"], output_dir / "authentication_statistics.csv"
            ),
            "track_statistics": write_track_statistics_csv(
                results["track_statistics"], output_dir / "track_statistics.csv"
            ),
        }

    def write_plots(self, output_dir: Path | str, pd_model: str, pd_params: dict, max_range_m: float) -> list:
        """Render all 9 required plots to `output_dir`."""
        return AnalysisPlotter(self.record).plot_all(output_dir, pd_model, pd_params, max_range_m)

    def write_engineering_report(self, output_dir: Path | str) -> Path:
        """Write one generated markdown summary of this run's metrics."""
        results = self.compute_all()
        metrics = results["performance_metrics"]
        roc = results["roc_curve"]
        identity_matrix = results["identity_confusion_matrix"]
        auth_matrix = results["authentication_confusion_matrix"]

        text = f"""# Phase 10 Analysis -- Run Engineering Report

Generated automatically from one `PipelineRunRecord` by
`AnalysisReportGenerator.write_engineering_report`. See
`iff_simulator/analysis/docs/` for the static architecture/metrics/
complexity documentation this report's numbers should be read alongside.

## Performance Metrics

| Metric | Value |
|---|---|
| Detection Probability | {metrics.detection_probability:.4f} |
| False Alarm Rate | {metrics.false_alarm_rate:.4f} |
| Authentication Success Rate | {metrics.authentication_success_rate:.4f} |
| Reply Success Rate | {metrics.reply_success_rate:.4f} |
| Decoder Success Rate | {metrics.decoder_success_rate:.4f} |
| Track Confirmation Rate | {metrics.track_confirmation_rate:.4f} |
| Average Track Lifetime (s) | {metrics.average_track_lifetime_s:.4f} |
| Average Detection Range (m) | {metrics.average_detection_range_m:.2f} |
| Maximum Detection Range (m) | {metrics.maximum_detection_range_m:.2f} |
| Average Processing Delay (us) | {metrics.average_processing_delay_us:.2f} |
| Average Propagation Delay (us) | {metrics.average_propagation_delay_us:.2f} |
| Average Receiver Delay (us) | {metrics.average_receiver_delay_us:.4f} |
| Average Track Update Delay (s) | {metrics.average_track_update_delay_s:.4f} |
| Average Total Delay (us) | {metrics.average_total_delay_us:.2f} |
| Average Signal Strength | {metrics.average_signal_strength:.4f} |
| Average SNR (dB, proxy) | {metrics.average_snr_db_proxy:.2f} |

## ROC

- Area Under Curve: {roc.area_under_curve:.4f}
- Points swept: {len(roc.thresholds)}

## Confusion Matrices

- Identity accuracy: {identity_matrix.accuracy:.4f}
- Authentication accuracy: {auth_matrix.accuracy:.4f}
"""
        output_path = Path(output_dir) / "engineering_report.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        return output_path
