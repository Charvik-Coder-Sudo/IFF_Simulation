# Phase 10 — Test Summary

## New test files (61 tests, all passing)

| File | Tests | Covers |
|---|---|---|
| `tests/test_analysis_performance_metrics.py` | 13 | Perfect detection, missed detections, no-reply-at-all, false alarm rate, authentication success rate (incl. zero-Mode-5 case), decoder success rate (garbled accounting), track confirmation rate, average track lifetime, detection range avg/max, processing/propagation/receiver/total delay, SNR proxy formula, empty-record zero-safety. |
| `tests/test_analysis_roc.py` | 8 | No-samples safety, perfect separability (AUC~=1), no separability (AUC~=0.5), monotonic TPR/FPR sweep, AUC in `[0,1]`, exclusion of signal-strength-less measurements, default threshold range, CSV row shape. |
| `tests/test_analysis_confusion_matrix.py` | 11 | Generic `ConfusionMatrix` precision/recall/F1/accuracy against hand-computed values, empty-pairs safety, perfect-prediction accuracy, identity vocabulary mapping (Ground Truth legacy strings + reported BLUE/RED/NEUTRAL/UNKNOWN), misclassification detection, NO_REPLY/GARBLED exclusion, false-alarm/fruited exclusion (no Ground Truth aircraft), authentication matrix via `AuthenticationEngine`, Mode S exclusion, mismatch detection, deterministic CSV. |
| `tests/test_analysis_latency.py` | 8 | Exact mean/min/max/stdev on synthetic delays, total-delay-matches-arrival-minus-time, documented-zero Scheduler/Track-Update delay, `receiver_delay_us` zero-case and nonzero-discrepancy-detection, `None` handling for missing timestamps, empty-record zero-safety, deterministic 6-row CSV. |
| `tests/test_analysis_statistics.py` | 11 | Numeric helpers (`mean`/`min_max`/`population_stdev`/`safe_divide`) on empty and populated inputs, per-mode detection statistics + empty-mode safety, per-level authentication statistics, completed-track and active-track `TrackStatistics` rows (partial-field behavior for active tracks), deterministic CSVs for all three. |
| `tests/test_analysis_report_generator.py` | 7 | Full pipeline integration: all 6 CSVs produced non-empty, all 9 plots produced non-empty, engineering report generated, same-seed byte-identical CSV determinism, different-seed metric divergence, Ground-Truth-never-mutated, `compute_all()` key completeness. |
| `tests/test_analysis_regression.py` | 3 | Core pipeline classes not shadowed/monkeypatched by the analysis package, Phase 9's all-off backward-compatibility invariant still holds with `iff_simulator.analysis` imported, `Scenario` exposes no mutation API. |

## Full suite result

```
512 passed, 1 skipped in ~24s
```

(The 1 skip is the pre-existing Phase 9 `@pytest.mark.slow` full-scale
300-aircraft x 30,000-tick performance test, intentionally excluded from
the default run — see `docs/REGRESSION_SUMMARY.md`.)

Before this phase's work began: **451 passed, 1 skipped** (452 total).
After: **512 passed, 1 skipped** (513 total) — a net addition of exactly
61 new tests, zero regressions, zero newly-skipped or newly-failing tests.
