# Phase 10 — Engineering Report

## What was built

A new, read-only `iff_simulator/analysis/` package (8 modules + this
`docs/` folder) that turns one completed pipeline run's output into:

- 16 scalar performance metrics (`performance_metrics.py`)
- A sensitivity-threshold ROC curve + AUC (`roc_analysis.py`)
- Identity (3x3) and Authentication (2x2) confusion matrices with
  precision/recall/F1/accuracy (`confusion_matrix.py`)
- A 6-component latency breakdown (`latency_analysis.py`)
- Per-mode/per-level/per-track statistics tables (`statistics.py`)
- 9 diagnostic plots (`plots.py`)
- 6 CSV files + 1 generated engineering-report markdown, all wired
  together by one composition root (`report_generator.py`)

Plus a new demo entry point (`run_analysis.py`, mirroring every prior
phase's `run_*.py` convention) and 61 new tests across 7 files.

## Example run (real Ground Truth scenario, `run_analysis.py`)

```
Interrogations analyzed: 29170
  Detection Probability:        0.9279
  False Alarm Rate:              0.0013
  Authentication Success Rate:   0.0000   (scenario has no Mode 5 traffic --
                                            DefaultModeSelectionPolicy always
                                            selects Mode S, see docs/METRICS.md)
  Reply Success Rate:            0.9279
  Decoder Success Rate:          0.9330
  Track Confirmation Rate:       0.0811
  Average Detection Range (m):   143.40
  Average Signal Strength:       0.9763
Saved 6 CSV outputs to iff_simulator/output/analysis
Saved 9 plots to iff_simulator/output/analysis_plots
Saved engineering report to iff_simulator/output/analysis/engineering_report.md
```

(Authentication Success Rate reads 0.0000 in this particular demo
scenario only because the loaded Ground Truth's aircraft never actually
get interrogated in Mode 5 -- `InterrogationScheduler`'s
`DefaultModeSelectionPolicy` always selects Mode S, a pre-existing
Phase 5 default unrelated to this phase. `test_analysis_statistics.py`
and `test_analysis_confusion_matrix.py` directly exercise the Mode 5
authentication paths with hand-built fixtures to cover this.)

## Key engineering judgment calls (all documented in `docs/METRICS.md`)

1. **Decoder Success Rate** uses `Replies_Received / (Replies_Received +
   Replies_Garbled)` — the only interpretation recoverable from existing
   aggregate `ReceiverStatistics` counters, since "reached the receiver
   but timed out" and "never reached the receiver at all" both surface
   identically as `NO_REPLY`.
2. **ROC analysis** is scoped to the receiver's `signal_strength`/
   sensitivity threshold specifically, not a general Pd-vs-Pfa sweep,
   because Phase 9's `ReceiverEffectsPipeline` only computes
   `signal_strength` for replies that already survived the Pd roll.
3. **Scheduler Delay** and **Track Update Delay** are documented
   constants (`0.0`), not computed approximations — the current
   architecture has no queuing/deferred-update stage to measure a
   nonzero value from.
4. **SNR** is an explicitly-labeled synthetic proxy
   (`10*log10(signal_strength/0.01)`), since no SNR concept exists
   anywhere else in this codebase.
5. **Identity confusion-matrix categories** (FRIENDLY/FOE/UNKNOWN) fold
   both Ground Truth's legacy identity vocabulary and the pipeline's
   BLUE/RED/NEUTRAL/UNKNOWN vocabulary into one consistent 3-category
   scheme, defined locally inside `confusion_matrix.py` rather than
   touching `authentication.py`'s existing mapping.

## Verification performed

- `python -m pytest tests/ -q` -> 512 passed, 1 skipped (see
  `docs/TEST_SUMMARY.md`).
- Every pre-existing `run_*.py` script re-run and confirmed unchanged
  (see `docs/REGRESSION_SUMMARY.md`).
- `python run_analysis.py` -> produces the 6 CSVs, 9 plots, and
  generated engineering report under `iff_simulator/output/analysis*`.
- Empirical raw-Pd-roll average cross-checked against the analytic
  `pd_gaussian` model at the run's average detection range, within
  0.02% (see `docs/VALIDATION_REPORT.md`).

## What was deliberately not done

- No pipeline module was modified (`GeometryEngine`, `Receiver`,
  `Propagation`, `Scheduler`, `TrackManager`, `Decoder`, Ground Truth —
  all untouched).
- No Kalman filtering, sensor fusion, or estimation of any kind was
  introduced (out of scope for this phase and for the project overall
  at this stage, per the standing project constraints).
- No new randomness was introduced — `iff_simulator.analysis` is
  entirely deterministic given its input `PipelineRunRecord` (whatever
  determinism the *run* had, from Phase 9's seeded RNG, is preserved
  and exposed, never added to).
