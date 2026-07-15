"""Ties a Scenario to a SimulationClock to expose a single "current" state.

Purpose:
    Implements `WorldState`, which advances a `SimulationClock` and
    keeps each aircraft's "current" `AircraftState` (in its `Scenario`)
    pointed at the recorded sample matching the clock's current time.

Inputs:
    A `Scenario` and a `SimulationClock`.

Outputs:
    `update()` advances the clock and refreshes the current-state
    cursor for every aircraft. `current_state()` returns a snapshot
    mapping aircraft_id -> its current `AircraftState`.

Engineering explanation:
    `WorldState` performs no motion propagation, geometry, or IFF
    logic — it only advances time and looks up which already-recorded
    sample corresponds to that time. This is the minimal scaffolding
    later phases need to drive a live playback loop over recorded
    ground truth, without adding any new simulation behavior now.
"""

from __future__ import annotations

import bisect

from ..domain import AircraftState, Scenario
from .clock import SimulationClock


class WorldState:
    """Advances simulated time and exposes each aircraft's current state.

    Purpose:
        Provide the single point where "simulated time" and "recorded
        aircraft state" meet: as the clock advances, `WorldState` keeps
        each aircraft's current-state cursor aligned with that time.

    Inputs:
        scenario: the `Scenario` holding aircraft and their recorded
            state histories.
        clock: the `SimulationClock` driving simulated time.

    Outputs:
        `update()` (advances time and refreshes cursors) and
        `current_state()` (a snapshot of every aircraft's current state).

    Engineering explanation:
        Looking up "the recorded sample at or after the current clock
        time" is a plain sorted-list lookup (`bisect`), not a physics
        computation — there is no dead-reckoning or interpolation here,
        matching the "no motion propagation" constraint for this phase.
    """

    def __init__(self, scenario: Scenario, clock: SimulationClock) -> None:
        self.scenario = scenario
        self.clock = clock

    def update(self) -> None:
        """Advance the clock by one step and refresh every aircraft's
        current-state cursor to match the new time."""
        self.clock.step()
        current_time = self.clock.current_time()
        for aircraft_id in self.scenario.list_aircraft_ids():
            history = self.scenario.get_state_history(aircraft_id)
            times = [state.time for state in history]
            index = bisect.bisect_left(times, current_time)
            index = min(index, len(history) - 1)
            self.scenario.set_current_index(aircraft_id, index)

    def current_state(self) -> dict[str, AircraftState]:
        """Return a snapshot of every aircraft's current `AircraftState`."""
        return {
            aircraft_id: self.scenario.get_state(aircraft_id)
            for aircraft_id in self.scenario.list_aircraft_ids()
        }
