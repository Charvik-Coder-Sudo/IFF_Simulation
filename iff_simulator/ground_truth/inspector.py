"""Read-only query interface over a Scenario's recorded state histories.

Purpose:
    Implements `GroundTruthInspector`, a convenience layer for asking
    common questions about a `Scenario` without every caller
    re-implementing the same filtering/aggregation logic over domain
    objects.

Inputs:
    A `Scenario`, as produced by `GroundTruthLoader.load()`.

Outputs:
    Scalars, lists of `AircraftState`, and dictionaries answering
    targeted questions (which aircraft exist, one aircraft's full
    recorded history, every aircraft's state at a given time, bounding
    boxes, durations, sample rates, path lengths).

Engineering explanation:
    This is a pure read-only, side-effect-free query layer over
    `Scenario`/`AircraftState` domain objects — no pandas DataFrame is
    ever accepted or returned by its public API. It never mutates the
    underlying Scenario, which lets multiple future modules (Ownship,
    Geometry, PSR, IFF, Scheduler, Receiver, Decoder) share one
    Inspector instance safely. `trajectory_length()` builds a small,
    private, throwaway DataFrame purely to reuse pandas/numpy's
    pairwise-summation algorithm over many segment lengths, matching
    the pre-refactor computation bit-for-bit; that DataFrame is never
    exposed to callers.
"""

from __future__ import annotations

import bisect

import numpy as np
import pandas as pd

from ..domain import AircraftState, Scenario


class GroundTruthInspector:
    """Answers common questions about a Scenario's recorded state histories.

    Purpose:
        Provide a single, reusable, read-only query API over a
        `Scenario`, so future modules never need to hand-roll
        filtering/aggregation over `AircraftState` histories for basic
        questions.

    Inputs:
        `scenario`: the `Scenario` produced by `GroundTruthLoader.load()`.

    Outputs:
        See individual method docstrings.

    Engineering explanation:
        Every method here is a pure function of the stored `Scenario` —
        no internal state is mutated, so the same Inspector instance can
        be safely reused and shared across modules. Method names and
        signatures are unchanged from the pre-refactor, DataFrame-based
        Inspector; only the underlying data source and return types
        (domain objects instead of DataFrame slices) have changed.
    """

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario

    def list_targets(self) -> list[str]:
        """Return the sorted list of distinct aircraft IDs present."""
        return sorted(self.scenario.list_aircraft_ids())

    def get_target(self, target_id: str) -> list[AircraftState]:
        """Return the full time-ordered recorded state history for one aircraft."""
        self._require_target(target_id)
        return self.scenario.get_state_history(target_id)

    def get_time(self, time: float) -> dict[str, AircraftState]:
        """Return the recorded state of every aircraft present at a given Time."""
        result: dict[str, AircraftState] = {}
        for target_id in self.list_targets():
            history = self.scenario.get_state_history(target_id)
            times = [state.time for state in history]
            index = bisect.bisect_left(times, time)
            if index < len(times) and times[index] == time:
                result[target_id] = history[index]
        return result

    def summary(self) -> dict:
        """Return a dict summarizing the whole dataset: target count,
        row count, and time range."""
        targets = self.list_targets()
        all_times = [
            state.time
            for target_id in targets
            for state in self.scenario.get_state_history(target_id)
        ]
        return {
            "target_count": len(targets),
            "targets": targets,
            "total_rows": len(all_times),
            "time_min": float(min(all_times)),
            "time_max": float(max(all_times)),
        }

    def bounding_box(self, target_id: str | None = None) -> dict:
        """Return the min/max X, Y, Z extent, for one target or the whole dataset."""
        states = self._states_for(target_id)
        xs = [state.position.x for state in states]
        ys = [state.position.y for state in states]
        zs = [state.position.z for state in states]
        return {
            "x_min": float(min(xs)),
            "x_max": float(max(xs)),
            "y_min": float(min(ys)),
            "y_max": float(max(ys)),
            "z_min": float(min(zs)),
            "z_max": float(max(zs)),
        }

    def duration(self, target_id: str | None = None) -> float:
        """Return the flight duration (max Time - min Time), for one target
        or the whole dataset."""
        states = self._states_for(target_id)
        times = [state.time for state in states]
        return float(max(times) - min(times))

    def sample_rate(self, target_id: str | None = None) -> float:
        """Return the constant timestep between consecutive samples, for one
        target (or the first target, if none is given)."""
        target_id = target_id or self.list_targets()[0]
        history = self.get_target(target_id)
        if len(history) < 2:
            return float("nan")
        return float(history[1].time - history[0].time)

    def trajectory_length(self, target_id: str) -> float:
        """Return the total 3D path length flown by one target, in the same
        distance units as X/Y/Z (typically meters).

        Uses a small internal (throwaway) DataFrame purely to perform the
        summation with the same pairwise-summation algorithm pandas/numpy
        used pre-refactor, so this stays numerically identical to the
        pre-refactor, DataFrame-based computation. The DataFrame never
        leaves this method.
        """
        history = self.get_target(target_id)
        positions = pd.DataFrame(
            {
                "X": [state.position.x for state in history],
                "Y": [state.position.y for state in history],
                "Z": [state.position.z for state in history],
            }
        )
        deltas = positions.diff().dropna()
        segment_lengths = np.sqrt((deltas**2).sum(axis=1))
        return float(segment_lengths.sum())

    def _states_for(self, target_id: str | None) -> list[AircraftState]:
        if target_id is not None:
            return self.get_target(target_id)
        return [
            state
            for tid in self.list_targets()
            for state in self.scenario.get_state_history(tid)
        ]

    def _require_target(self, target_id: str) -> None:
        if target_id not in self.scenario.list_aircraft_ids():
            raise KeyError(f"Unknown TargetID: {target_id}")
