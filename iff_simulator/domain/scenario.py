"""Container for every aircraft and its recorded state history.

Purpose:
    Hold the complete set of `Aircraft` metadata objects and their
    associated recorded `AircraftState` history for one simulation run,
    replacing the old "dict of DataFrames" as the object passed between
    modules.

Inputs:
    A list of `Aircraft` and a mapping of aircraft_id -> time-ordered
    list of `AircraftState`, as built by `GroundTruthLoader`.

Outputs:
    Read accessors used by every downstream module (`GroundTruthValidator`,
    `GroundTruthMerger`, `GroundTruthInspector`, `WorldState`).

Engineering explanation:
    A `Scenario` keeps the full recorded time history per aircraft (so
    Phase 1's statistics/plots, which need the entire trajectory, keep
    working) while also tracking a "current index" per aircraft — a
    cursor into that history — so a future live playback driven by
    `SimulationClock`/`WorldState` can expose a single "current"
    `AircraftState` per aircraft without duplicating the recorded data.
"""

from __future__ import annotations

from .aircraft import Aircraft
from .aircraft_state import AircraftState


class Scenario:
    """The set of aircraft and their recorded state histories for one run.

    Purpose:
        Single, reusable domain object representing "everything ground
        truth knows about this run": which aircraft exist, and what
        each one's full recorded trajectory looks like.

    Inputs:
        aircraft: list of `Aircraft` metadata objects.
        state_history: dict mapping aircraft_id -> time-ordered list of
            `AircraftState`.

    Outputs:
        `get_aircraft()`, `get_all_aircraft()`, `get_state()`,
        `get_state_history()`, `list_aircraft_ids()`.

    Engineering explanation:
        Aircraft order is preserved from the input list (matching file
        discovery order), while state history is looked up by
        aircraft_id for O(1) access. No pandas/DataFrame type appears
        anywhere in this class's public surface.
    """

    def __init__(
        self,
        aircraft: list[Aircraft],
        state_history: dict[str, list[AircraftState]],
    ) -> None:
        self._aircraft_order: list[str] = [a.aircraft_id for a in aircraft]
        self._aircraft_by_id: dict[str, Aircraft] = {a.aircraft_id: a for a in aircraft}
        self._state_history: dict[str, list[AircraftState]] = state_history
        self._current_index: dict[str, int] = {aircraft_id: 0 for aircraft_id in state_history}

    def list_aircraft_ids(self) -> list[str]:
        """Return aircraft IDs in their original discovery order."""
        return list(self._aircraft_order)

    def get_aircraft(self, aircraft_id: str) -> Aircraft:
        """Return the `Aircraft` metadata object for one aircraft ID."""
        if aircraft_id not in self._aircraft_by_id:
            raise KeyError(f"Unknown aircraft_id: {aircraft_id}")
        return self._aircraft_by_id[aircraft_id]

    def get_all_aircraft(self) -> list[Aircraft]:
        """Return every `Aircraft` metadata object, in discovery order."""
        return [self._aircraft_by_id[aircraft_id] for aircraft_id in self._aircraft_order]

    def get_state_history(self, aircraft_id: str) -> list[AircraftState]:
        """Return the full time-ordered recorded state history for one aircraft."""
        if aircraft_id not in self._state_history:
            raise KeyError(f"Unknown aircraft_id: {aircraft_id}")
        return self._state_history[aircraft_id]

    def get_state(self, aircraft_id: str) -> AircraftState:
        """Return the "current" `AircraftState` for one aircraft.

        The current state is a cursor into that aircraft's recorded
        history, advanced by `WorldState.update()`. It defaults to the
        first recorded sample.
        """
        history = self.get_state_history(aircraft_id)
        index = self._current_index[aircraft_id]
        return history[index]

    def set_current_index(self, aircraft_id: str, index: int) -> None:
        """Advance the "current state" cursor for one aircraft.

        Used internally by `WorldState.update()`; not needed by Phase 1's
        static (non-live) ground-truth reporting pipeline.
        """
        history = self.get_state_history(aircraft_id)
        if not 0 <= index < len(history):
            raise IndexError(
                f"Index {index} out of range for aircraft '{aircraft_id}' "
                f"with {len(history)} recorded states."
            )
        self._current_index[aircraft_id] = index
