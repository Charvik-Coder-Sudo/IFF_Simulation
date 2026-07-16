# Phase 10 — Algorithmic Complexity

Let `n` = number of interrogations/tick_results in a `PipelineRunRecord`,
`t` = number of active + completed tracks, `k` = number of ROC
threshold points (default 50, a fixed constant independent of `n`).

| Module / function | Complexity | Notes |
|---|---|---|
| `statistics.mean` / `population_stdev` / `min_max` / `safe_divide` | O(m) / O(1) | `m` = length of the input list; `safe_divide` is O(1). |
| `performance_metrics.compute_performance_metrics` | O(n) | A fixed number of single-pass filters/reductions over `tick_results`; no nested loop over `n`. |
| `roc_analysis.compute_roc_curve` | O(n + k·n) = O(k·n), k constant -> **O(n)** | Building the `(signal_strength, is_real)` list is O(n); each of the `k` thresholds does an O(n) count. Since `k` is a fixed constant (50 by default), this is O(n) overall, not O(n^2) in any growing dimension. The final AUC step sorts the k points: O(k log k), negligible. |
| `confusion_matrix.compute_confusion_matrix` | O(m + L^2) | `m` = number of pairs, `L` = number of labels (3 or 2, a small constant) -> effectively **O(m)**. |
| `confusion_matrix.compute_identity_confusion_matrix` / `compute_authentication_confusion_matrix` | O(n) | One pass over `tick_results`, one `Scenario.get_aircraft` (O(1) dict lookup, per `scenario.py`) per real `VALID` measurement. |
| `latency_analysis.compute_latency_breakdown` | O(n) | A fixed number of single-pass filters/reductions over `tick_results`. |
| `statistics.compute_detection_statistics` | O(n · M) = **O(n)**, M constant (3 modes + 1 "ALL") | Each of the 4 rows does one O(n) pass; M is a fixed constant (`IFFMode` has exactly 3 members). |
| `statistics.compute_authentication_statistics` | O(n) | Same pattern, 3 fixed rows. |
| `statistics.compute_track_statistics` | O(t) | One row per completed + active track, each O(1) to build. |
| `plots.AnalysisPlotter.plot_*` | O(n) each (or O(t) for track-based plots) | Matplotlib's own rendering cost for a fixed-size figure is O(number of points plotted), i.e. O(n)/O(t); no per-plot algorithm is worse than linear in its own data size. |
| `report_generator.AnalysisReportGenerator.compute_all` / `write_csv_outputs` / `write_plots` | O(n + t) | Sum of the above; every sub-computation is linear (or effectively constant-factor linear) in run size. |

**No metric in this package is worse than O(n log n)** (the only
`log`-factor anywhere is the trivial `sort` of `k=50` ROC points, a
fixed constant, not a function of `n`). This matches the codebase's
existing complexity discipline (e.g. `World.step()`'s O(log n) bisect
lookup, `ReceiverBuffer`'s O(log n) heap) — analysis over one run's
output is comfortably cheaper than the O(n log n) simulation that
produced it.
