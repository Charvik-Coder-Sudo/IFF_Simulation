# Phase 10 — Analysis Package Architecture

## Scope

`iff_simulator/analysis/` is a **read-only** layer on top of the
existing, unmodified simulator pipeline (Phases 1-9). It consumes
already-produced pipeline objects and computes performance metrics, ROC
curves, confusion matrices, latency breakdowns, CSVs, and plots. It
never calls a mutator on `Scenario`/`Aircraft`/`AircraftState`, never
recomputes geometry, and never re-implements any logic that already
exists in `GeometryEngine`, `Receiver`, `Propagation`, `Scheduler`,
`TrackManager`, or `Decoder`.

## Data flow

```mermaid
flowchart TD
    subgraph Existing Pipeline (Phases 1-9, unmodified)
        GT[Ground Truth / Scenario]
        GE[GeometryEngine]
        TS[TargetSelector]
        SCH[InterrogationScheduler]
        TXP[AirborneTransponder]
        REP[ReceiverEffectsPipeline<br/>Propagation, Pd, Receiver,<br/>Garbling, Fruiting, Reply Loss, Decoder]
        TM[IFFTrackManager]
    end

    GT --> GE --> TS --> SCH --> TXP --> REP --> TM

    subgraph Caller (run_analysis.py / a test)
        REC[PipelineRunRecord<br/>scenario, interrogations, replies,<br/>tick_results, active_tracks,<br/>completed_track_summaries,<br/>receiver_statistics]
    end

    SCH -.captured.-> REC
    TXP -.captured.-> REC
    REP -.captured.-> REC
    TM -.captured.-> REC
    GT -.read-only identity lookups.-> REC

    subgraph iff_simulator.analysis (Phase 10, new)
        PM[performance_metrics.py]
        ROC[roc_analysis.py]
        CM[confusion_matrix.py]
        LAT[latency_analysis.py]
        ST[statistics.py]
        PL[plots.py]
        RG[report_generator.py]
    end

    REC --> PM
    REC --> ROC
    REC --> CM
    REC --> LAT
    REC --> ST
    REC --> PL
    PM --> RG
    ROC --> RG
    CM --> RG
    LAT --> RG
    ST --> RG
    PL --> RG

    RG --> CSVs[6 CSV files]
    RG --> PNGs[9 PNG plots]
    RG --> ENGRPT[engineering_report.md]
```

## Key design decision: `PipelineRunRecord`

`PipelineRunRecord` (`run_record.py`) is the *only* new wiring this
phase introduces, and it is a plain data container, not a pipeline
stage: every field is something a caller already has on hand after
driving the existing pipeline (see `run_analysis.py` for the reference
wiring, mirroring `run_receiver_pipeline.py`'s own conventions). No
pipeline class constructs or depends on a `PipelineRunRecord` --
analysis always happens strictly after a run has completed.

## Module responsibilities

| Module | Responsibility |
|---|---|
| `run_record.py` | `PipelineRunRecord` — the one input bundle. |
| `statistics.py` | Numeric helpers (`mean`, `population_stdev`, `min_max`, `safe_divide`) + per-mode/per-level/per-track breakdown rows. |
| `performance_metrics.py` | The 16 scalar metrics (Pd, Pfa, auth/reply/decoder success rates, track confirmation, ranges, delays, signal strength, SNR proxy). |
| `roc_analysis.py` | ROC curve over the receiver's `signal_strength`/sensitivity threshold. |
| `confusion_matrix.py` | Identity (3x3) and Authentication (2x2) confusion matrices + precision/recall/F1/accuracy. |
| `latency_analysis.py` | Six-component latency breakdown. |
| `plots.py` | 9 diagnostic PNGs. |
| `report_generator.py` | Composition root: computes everything, writes every CSV/plot/report. |

## Why this design, not an alternative

An alternative would have been to add analysis hooks directly inside
`ReceiverEffectsPipeline`/`IFFTrackManager` (e.g. an `on_tick` callback).
That was rejected: it would touch completed, tested modules for a
capability that is purely diagnostic and post-hoc, violating this
phase's explicit "do not modify" constraint for no real benefit — a
plain data container passed to a separate package achieves the same
analytical power with zero coupling in either direction.
