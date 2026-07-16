# Phase 10 — Metric Definitions and Mathematical Derivation

All formulas below are implemented in `performance_metrics.py`,
`roc_analysis.py`, `confusion_matrix.py`, `latency_analysis.py`, and
`statistics.py`. Every ratio uses `safe_divide(numerator, denominator)`
(`statistics.py`), which returns `0.0` for a zero denominator instead of
raising or producing `NaN` — every metric below is well-defined even on
an empty or degenerate run.

## Core performance metrics (`performance_metrics.py`)

### Detection Probability (Pd)

```
Pd = Correct Replies / Expected Replies
```

- **Expected Replies**: ticks where the transponder actually produced a
  `ReplyMessage` (`record.replies[i] is not None`) — i.e. the target
  *tried* to reply.
- **Correct Replies**: of those, ticks whose `real_measurement.reply_status
  == VALID` — i.e. the reply survived the Pd roll, sensitivity
  threshold, garbling, and timeout, and was cleanly decoded.

This empirically measures how well the receiver's actual behavior
tracks the configured `pd_model` in `ReceiverConfig` (see
`detection.py`'s `pd_gaussian`/`pd_inverse_quartic`/`pd_always_detect`) —
`docs/VALIDATION_REPORT.md` cross-checks this.

### False Alarm Rate (Pfa, simple ratio)

```
Pfa = False Replies / (Replies Received + False Replies + Fruited Replies)
```

Taken directly from `ReceiverStatistics` (Phase 9's own aggregate
counters). This is a simple "what fraction of everything the receiver
reported was a false alarm" ratio — distinct from the ROC's False
Positive Rate (see below), which is a threshold-swept discrimination
metric over a different (and narrower) universe.

### Authentication Success Rate

```
Authentication Success Rate = Authenticated / Mode 5 Replies
```

Both counted over `VALID` measurements with `mode in (MODE5_L1,
MODE5_L2)`; `Authenticated` = `authentication_status == AUTHENTICATED`.
Mode S measurements are excluded entirely (no authentication concept —
`derive_authentication_status` always assigns `NOT_APPLICABLE` for Mode S).

### Reply Success Rate

```
Reply Success Rate = Replies Received / Interrogations Sent
```

`Replies Received` = ticks whose `real_measurement.reply_status ==
VALID`; `Interrogations Sent` = `len(record.interrogations)`.

### Decoder Success Rate

```
Decoder Success Rate = Replies Received / (Replies Received + Replies Garbled)
```

**Engineering judgment call**: "received" (reached a decode attempt) vs
"lost before ever reaching the receiver" (Pd roll or sensitivity
rejection) are not separably recoverable from `DecodedIFFMeasurement`
alone — both surface as `NO_REPLY`. The honest, available-data-only
interpretation: restrict the denominator to outcomes we know reached a
decode attempt (`VALID` or `GARBLED`), and ask what fraction decoded
cleanly.

### Track Confirmation Rate

```
Track Confirmation Rate = Tracks Ever Confirmed / Total Tracks
```

`Total Tracks` = completed (`TrackSummary`) + still-active (`IFFTrack`)
tracks. "Ever confirmed": for completed tracks, `confirmed_time > 0.0`
(if a track spent nonzero time in the `CONFIRMED` state before being
lost, it was confirmed); for active tracks, `track_status == CONFIRMED`
right now.

### Average Track Lifetime

```
Average Track Lifetime = mean(TrackSummary.duration for completed tracks)
```

Computed **only** over completed (lost) tracks — an active track's
lifetime is right-censored (still ongoing, no defined endpoint yet), so
including it would understate true lifetime, not just add noise.

### Average / Maximum Detection Range

```
Average Detection Range = mean(range_m for VALID measurements)
Maximum Detection Range = max(range_m for VALID measurements)
```

### Average Processing / Propagation Delay

Straight means of `DecodedIFFMeasurement.processing_delay` /
`.propagation_delay` (already microseconds) over `VALID` measurements.

### Average Receiver Delay

```
Receiver Delay = arrival_time - (time + (processing_delay + propagation_delay) / 1e6)
```
(converted to microseconds). **This is expected to be ~0** in the
current architecture: Phase 9's `ReceiverEffectsPipeline` computes
`arrival_time` as *exactly* this sum (see `propagation.py`'s
`propagate()` and `jitter.py`'s `JitteredReplyPropagation`) — there is
no receiver-internal latency beyond propagation timing to measure yet.
This metric exists for schema completeness and as a regression
detector: if a future phase *does* introduce receiver-internal latency,
this metric will pick it up with zero changes to this analysis code.

### Average Track Update Delay

**Documented constant `0.0`.** `IFFTrackManager.update()` is always
called synchronously in the same call stack as decoding the measurement
that feeds it (see every `run_*.py` script's main loop) — there is no
deferred/batched track update in the current architecture to measure a
nonzero delay from.

### Average Total (End-to-End) Delay

```
Total Delay = (arrival_time - time) * 1e6   [microseconds]
```
Cross-checked against `processing_delay + propagation_delay` for
consistency (they should match exactly, absent timing jitter that
changes `arrival_time` after the fact but not the reported delay
fields — see `test_analysis_latency.py`).

### Average Signal Strength

Mean `signal_strength` over every measurement (real, false-alarm, or
fruited) that has one (i.e. reached propagation).

### Average SNR (dB, proxy) — optional, clearly synthetic

```
SNR_proxy_dB = 10 * log10(average_signal_strength / NOISE_FLOOR)
```
`NOISE_FLOOR = 0.01` (`performance_metrics.py`) is a **documented
constant, not a measured quantity** — no SNR concept exists anywhere
else in this codebase (confirmed by exhaustive grep during planning).
This mirrors how `propagation.compute_signal_strength` is itself
already documented as "not a real dBm/Watt measurement."

## ROC Analysis (`roc_analysis.py`)

**Scope, precisely**: among every measurement (real, false-alarm, or
fruited) that reached propagation (has a non-`None` `signal_strength`),
sweep a threshold `t` and ask "how well would a `signal_strength >= t`
rule separate real transponder replies from false/fruited noise":

```
TPR(t) = |{real replies with signal_strength >= t}| / |{all real replies with known signal_strength}|
FPR(t) = |{false/fruited with signal_strength >= t}| / |{all false/fruited with known signal_strength}|
AUC = trapezoidal_integral(TPR over FPR, points sorted by ascending FPR)
```

**Why this scope, not a general Pd-vs-Pfa curve**: `ReceiverEffectsPipeline
._submit_real` (Phase 9) returns *before* calling `propagation.propagate()`
whenever the Pd roll fails — so a Pd-rejected reply never has a
`signal_strength` value at all. A classical full-sweep ROC over every
*attempted* reply is therefore not reconstructable post-hoc without
modifying the pipeline (forbidden by this phase's constraints). This
ROC instead characterizes exactly the parameter Phase 9 actually
exposes for this purpose: `ReceiverConfig.sensitivity_threshold`.

Default thresholds: 50 points evenly spaced between the minimum and
maximum observed `signal_strength` this run (falls back to `[0, 1]` if
no samples exist).

## Confusion Matrices (`confusion_matrix.py`)

### Generic `ConfusionMatrix`

For a fixed label set and `(true_label, predicted_label)` pairs, per
label `L`:
```
TP(L) = counts[L][L]
FP(L) = sum(counts[other][L] for other != L)
FN(L) = sum(counts[L][other] for other != L)
Precision(L) = TP / (TP + FP)
Recall(L)    = TP / (TP + FN)
F1(L)        = 2 * Precision * Recall / (Precision + Recall)
Accuracy     = sum(counts[L][L] for all L) / total pairs
```

### Identity matrix (3x3: FRIENDLY / FOE / UNKNOWN)

**Engineering judgment call**: Ground Truth (`Aircraft.identity`, a
legacy string) and the pipeline's reported identity
(`DecodedIFFMeasurement.identity`, BLUE/RED/NEUTRAL/UNKNOWN) use
different vocabularies from each other and from the spec's requested
3-category matrix. `confusion_matrix.py` defines its own local
`_to_identity_class` mapping (FRIEND/FRIENDLY/BLUE -> FRIENDLY;
FOE/HOSTILE/ENEMY/RED -> FOE; anything else, including NEUTRAL -> UNKNOWN),
applied identically to both sides. Only `VALID` real-origin measurements
are included — false-alarm/fruited measurements have no Ground Truth
aircraft to compare against and are excluded.

### Authentication matrix (2x2: AUTHENTICATED / FAILED)

Ground truth label = `AuthenticationEngine().authenticate(scenario.get_aircraft(target_id))`
(the exact same, already-tested logic `Mode5ReplyGenerator` itself uses
— never reimplemented). Reported label =
`measurement.authentication_status == AUTHENTICATED`. Restricted to
Mode 5 `VALID` measurements only (Mode S has no authentication concept).

## Latency Analysis (`latency_analysis.py`)

Six components, each reported as mean/min/max/population-stdev:
Scheduler Delay (documented `0.0` — synchronous transmission, no queuing
delay exists in this architecture), Processing Delay, Propagation
Delay, Receiver Delay (see formula above), Track Update Delay
(documented `0.0` — synchronous update), Total End-to-End Delay.

## Detection / Authentication / Track statistics (`statistics.py`)

Per-category breakdown tables mirroring the scalar metrics above but
split by `IFFMode` (detection), Mode 5 level (authentication), or
individual track (track statistics) — see each dataclass's own
docstring for its exact row schema. `TrackStatistics` rows for
still-active tracks necessarily have fewer populated fields than
completed-track rows (see that module's docstring): `IFFTrack` never
accumulated the lifetime counters `TrackSummary` only computes at loss
time.
