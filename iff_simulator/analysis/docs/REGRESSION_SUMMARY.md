# Phase 10 — Regression Summary

## Full suite, before vs after

| | Passed | Skipped | Total |
|---|---|---|---|
| Before Phase 10 (Phase 1-9 complete) | 451 | 1 | 452 |
| After Phase 10 | 512 | 1 | 513 |
| Delta | +61 | 0 | +61 |

Zero pre-existing tests were modified, removed, or newly skipped/failed.
The single skip (a `@pytest.mark.slow` full-scale 300-aircraft x
30,000-tick performance test) is a pre-existing Phase 9 test, unrelated
to this phase, unaffected by it.

## Entry-point scripts re-run (all succeed, unchanged behavior)

| Script | Result |
|---|---|
| `main.py` | Ground truth merge/statistics unchanged (6 targets, same distances/durations). |
| `run_world_simulation.py` | Same alive/IFF-capable target lists, same final Ownship position. |
| `run_interrogation_scheduler.py` | 29,170 interrogations transmitted (unchanged). |
| `run_receive_pipeline.py` | 29,170/29,170 measurements, all VALID, sequence ordering preserved (unchanged). |
| `run_track_manager.py` | 29,170 track snapshots, 2 confirmed active tracks, 0 lost (unchanged). |
| `run_receiver_pipeline.py` (Phase 9) | Runs cleanly; produces `receiver_statistics.csv` + 7 plots as before. |
| `run_analysis.py` (Phase 10, new) | Runs cleanly; produces the 6 required CSVs + 9 plots + engineering report. |

## Why zero regression was structurally guaranteed, not just tested

1. `iff_simulator/analysis/` is a brand-new package — adding a new
   package cannot change any existing module's behavior by itself.
2. The only modification to an existing file was purely additive: none
   were needed for Phase 10 (Phase 9's `csv_logging.py`/`__init__.py`
   additions predate this phase). Phase 10 touches **zero** existing
   files inside `iff_simulator/sensors/iff/`, `iff_simulator/ground_truth/`,
   `iff_simulator/geometry/`, `iff_simulator/domain/`, or
   `iff_simulator/simulation/`.
3. `PipelineRunRecord` and every `iff_simulator.analysis` function only
   ever *read* from pipeline output objects (`DecodedIFFMeasurement`,
   `IFFTrack`, `TrackSummary`, `ReceiverStatistics`, `Scenario.get_aircraft`)
   — never call a mutator, never monkeypatch, never reach into
   `GeometryEngine`/`Receiver`/`Propagation`/`Scheduler`/`TrackManager`/
   `Decoder` internals.
4. `test_analysis_regression.py` makes this structural guarantee an
   executable, ongoing check rather than a one-time claim.
