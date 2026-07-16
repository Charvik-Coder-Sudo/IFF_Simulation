"""The single input bundle every Phase 10 analysis module consumes.

Purpose:
    Defines `PipelineRunRecord`, a plain data container holding one
    completed run's worth of existing pipeline outputs. This is not a
    new simulation stage -- every field is something a caller already
    has on hand after driving `World` -> `TargetSelector` ->
    `InterrogationScheduler` -> `AirborneTransponder` ->
    `ReceiverEffectsPipeline` -> `IFFTrackManager` exactly as
    `run_receiver_pipeline.py` / `tests/test_receiver_pipeline.py`
    already do. Phase 10 adds no new simulation logic whatsoever; this
    class exists purely so the analysis package has one object to pass
    around instead of six separate parameters.

Inputs:
    Built by a caller (a script or test) after running the existing
    pipeline; never constructed by any pipeline stage itself.

Outputs:
    Consumed by every `iff_simulator.analysis` module.

Engineering explanation:
    `scenario` is retained purely for read-only Ground Truth lookups
    (`scenario.get_aircraft(target_id).identity`, and via
    `AuthenticationEngine` for the ground-truth authentication label) --
    no analysis code ever calls a `Scenario` mutator. `interrogations`
    and `replies` are parallel lists (index i's reply is the transponder
    response, or None, to `interrogations[i]`); `tick_results` is a
    third parallel list holding that same tick's full
    `ReceiverTickResult` (real + false-alarm + fruited measurements).
    Frozen, like every other per-run record in this codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..domain import Scenario
from ..sensors.iff import (
    IFFTrack,
    InterrogationMessage,
    ReceiverStatistics,
    ReceiverTickResult,
    ReplyMessage,
    TrackSummary,
)


@dataclass(frozen=True, slots=True)
class PipelineRunRecord:
    """Everything one completed pipeline run produced, for analysis.

    Purpose:
        Bundle the existing pipeline's output objects so every
        `iff_simulator.analysis` module has a single, uniform input
        shape.

    Inputs:
        scenario: read-only; only `.get_aircraft(...)` is ever called
            on it by analysis code.
        interrogations: every `InterrogationMessage` issued this run, in
            order (skips ticks where the scheduler produced no
            interrogation).
        replies: parallel to `interrogations` -- `replies[i]` is the
            `ReplyMessage` the transponder produced for
            `interrogations[i]`, or `None` if it didn't reply at all.
        tick_results: parallel to `interrogations` -- the
            `ReceiverTickResult` `ReceiverEffectsPipeline.process_tick`
            produced that same tick.
        active_tracks: `IFFTrackManager.get_active_tracks()` at the end
            of the run.
        completed_track_summaries:
            `IFFTrackManager.get_completed_track_summaries()`.
        receiver_statistics: `ReceiverEffectsPipeline.statistics.snapshot()`.

    Outputs:
        Consumed by every module in `iff_simulator.analysis`.
    """

    scenario: Scenario
    interrogations: list[InterrogationMessage] = field(default_factory=list)
    replies: list[ReplyMessage | None] = field(default_factory=list)
    tick_results: list[ReceiverTickResult] = field(default_factory=list)
    active_tracks: list[IFFTrack] = field(default_factory=list)
    completed_track_summaries: list[TrackSummary] = field(default_factory=list)
    receiver_statistics: ReceiverStatistics | None = None
