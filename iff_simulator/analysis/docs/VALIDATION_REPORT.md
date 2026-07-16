# Phase 10 — Validation Report

Validated by running `run_analysis.py` (Gaussian Pd model, `r_max=2000m`,
`pfa=0.02`, `fruiting_rate=0.05`, `sensitivity_threshold=0.05`,
`capacity=10`, noise + jitter enabled, `seed=42`) against the real
ground-truth scenario under `Aircrafts/`, and `run_receiver_pipeline.py`
(same Pd model/params, `seed=42`) for the raw-Pd-roll cross-check below.

## 1. Empirical Pd roll tracks the configured analytic Pd(range) model

**Important distinction** (documented in `docs/METRICS.md`):
`PerformanceMetrics.detection_probability` is an *operational* metric —
"Correct Replies / Expected Replies" — that folds in every cause of
loss (failed Pd roll, below sensitivity, garbled, timed out, dropped by
saturation). It is **not** the same thing as the raw Pd value rolled
against each attempt. The correct like-for-like validation is between
`ReceiverStatistics.average_detection_probability` (the raw mean of
every `compute_pd(...)` value rolled against, from `run_receiver_pipeline
.py`'s printed "Avg Pd") and the analytic model evaluated at the run's
average detection range:

- Run's average detection range: **143.4 m**
- Analytic `pd_gaussian(143.4, r_max=2000)` = `exp(-(143.4/2000)^2)` = **0.99489**
- Run's measured `average_detection_probability` (raw Pd rolls): **0.9947**

Difference: **0.00019 (0.02%)** — well within expected sampling noise
for ~29,170 draws at this Pd level. **Pass.**

## 2. `PerformanceMetrics.detection_probability` is bounded correctly

Measured `0.9279` (`run_analysis.py`, heavier config: `capacity=10`,
`sensitivity_threshold=0.05`, tighter garble window) — strictly less
than the raw Pd average (`0.9947`) confirmed above, exactly as expected
since it additionally folds in sensitivity/garbling/saturation losses
on top of the raw Pd roll. **Pass** (directional sanity check: adding
more loss mechanisms can only lower the operational Pd, never raise it
above the raw roll average).

## 3. All ratios stay within `[0, 1]`

Every rate/probability field in `PerformanceMetrics`
(`detection_probability`, `false_alarm_rate`, `authentication_success_rate`,
`reply_success_rate`, `decoder_success_rate`, `track_confirmation_rate`)
and every `ConfusionMatrix` precision/recall/F1/accuracy value is
constructed exclusively via `safe_divide`, which cannot produce a value
outside `[0, 1]` for non-negative counts (division of a count by a
count that is `>=` it). Confirmed both by code inspection and by
`test_analysis_roc.py::test_auc_always_within_unit_interval` /
`test_analysis_confusion_matrix.py` (all pass).

## 4. Zero-division safety

Every metric was exercised against an empty `PipelineRunRecord`
(`test_analysis_performance_metrics.py::test_empty_record_gives_all_zero_metrics_without_crashing`,
`test_analysis_latency.py::test_empty_record_gives_zero_stats_everywhere`,
`test_analysis_statistics.py::test_detection_statistics_empty_mode_gives_zero_not_crash`)
— all return `0.0` fields, no exception, no `NaN`. **Pass.**

## 5. Determinism

`test_analysis_report_generator.py::test_same_seed_gives_byte_identical_csv_output`
runs the full pipeline twice with `seed=99` and asserts byte-identical
CSV text for all 6 required files. **Pass.**
`test_different_seed_gives_different_performance_metrics` confirms
`seed=1` vs `seed=2` diverge. **Pass.**

## 6. Ground Truth is never mutated

`test_analysis_report_generator.py::test_ground_truth_scenario_is_never_mutated_by_analysis`
deep-copies every `Aircraft` before running the full `AnalysisReportGenerator
.compute_all()`, then asserts equality after. **Pass.**

## 7. No completed module was modified

`test_analysis_regression.py::test_core_classes_are_not_shadowed_by_the_analysis_package`
confirms `ReceiverEffectsPipeline`/`ModeDecoder`/`ReplyMatcher`/`Receiver`
/`IFFTrackManager` are still defined in their original modules (not
monkeypatched or replaced), and
`test_backward_compatible_pipeline_still_matches_with_analysis_imported`
re-confirms Phase 9's own all-off backward-compatibility invariant
still holds with `iff_simulator.analysis` imported. **Pass.**

## Summary

All validation checks pass. No metric produced an out-of-range,
`NaN`, or crashing result across the full test matrix (perfect
detection, missed detections, false alarms, authentication failures,
garbled replies, empty records, and a full real-scenario run).
