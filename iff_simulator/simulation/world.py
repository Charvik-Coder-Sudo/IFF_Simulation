"""The runtime world model: Scenario + SimulationClock + Ownship.

Purpose:
    Implements `World`, which owns a `Scenario`, a `SimulationClock`,
    and an `Ownship`, and advances simulated time one step at a time.
    This is the runtime object every future sensor/IFF module will be
    handed, instead of reaching into `Scenario`/`SimulationClock`
    separately.

Inputs:
    A `Scenario` and a `SimulationClock`, plus the aircraft_id of the
    Scenario aircraft designated as Ownship (Phase 2 designates Target 1).

Outputs:
    `step()` advances simulated time. `current_time()`,
    `get_targets()`, `get_target(id)`, `alive_targets()`, and
    `iff_capable_targets()` answer questions about the world's current
    state.

Engineering explanation:
    `World` performs no motion propagation, geometry, or IFF logic — it
    only advances the clock and looks up which already-recorded sample
    corresponds to that time (for every aircraft, including Ownship's
    own). "Targets" here means every Scenario aircraft other than the
    one designated as Ownship: once an aircraft becomes Ownship, it is
    "self," not something to be queried as a target.
"""

from __future__ import annotations

import bisect

from ..domain import Aircraft, Scenario
from ..sensors.iff import Ownship
from .clock import SimulationClock


class World:
    """Owns Scenario, SimulationClock, and Ownship; advances simulated time.

    Purpose:
        Single runtime object tying together "what time is it,"
        "what's the full recorded scenario," and "where is Ownship,"
        plus a query API for the other (non-Ownship) aircraft in the
        scenario.

    Inputs:
        scenario: the `Scenario` holding every aircraft and its
            recorded state history.
        clock: the `SimulationClock` driving simulated time.
        ownship_id: which Scenario aircraft is Ownship. Defaults to
            "TARGET_1", per Phase 2's designation of Target 1 as
            Ownship.
        **ownship_kwargs: forwarded to `Ownship` (pitch, roll,
            maximum_range, beam_width, interrogation_rate,
            operating_modes).

    Outputs:
        `step()`, `current_time()`, `get_targets()`, `get_target(id)`,
        `alive_targets()`, `iff_capable_targets()`.

    Engineering explanation:
        Looking up "the recorded sample at or after the current clock
        time" is a plain sorted-list lookup (`bisect`), not a physics
        computation — there is no dead-reckoning or interpolation here,
        matching the "no motion propagation" constraint for this phase.
    """

    def __init__(
        self,
        scenario: Scenario,
        clock: SimulationClock,
        ownship_id: str = "TARGET_1",
        **ownship_kwargs,
    ) -> None:
        self.scenario = scenario
        self.clock = clock
        self.ownship = Ownship(aircraft_id=ownship_id, scenario=scenario, **ownship_kwargs)
        # Recorded times per aircraft, extracted once so step() can binary
        # search each aircraft's already-sorted times in O(log n) instead of
        # rebuilding an O(n) list from AircraftState objects on every step.
        self._times_by_aircraft: dict[str, list[float]] = {
            aircraft_id: [state.time for state in scenario.get_state_history(aircraft_id)]
            for aircraft_id in scenario.list_aircraft_ids()
        }

    def step(self) -> None:
        """Advance the clock by one step and refresh every aircraft's
        (including Ownship's) current-state cursor to match the new time."""
        self.clock.step()
        current_time = self.clock.current_time()
        for aircraft_id, times in self._times_by_aircraft.items():
            index = bisect.bisect_left(times, current_time)
            index = min(index, len(times) - 1)
            self.scenario.set_current_index(aircraft_id, index)

    def current_time(self) -> float:
        """Return the world's current simulated time."""
        return self.clock.current_time()

    def get_targets(self) -> list[Aircraft]:
        """Return every Scenario aircraft other than Ownship."""
        return [
            self.scenario.get_aircraft(aircraft_id)
            for aircraft_id in self.scenario.list_aircraft_ids()
            if aircraft_id != self.ownship.aircraft_id
        ]

    def get_target(self, aircraft_id: str) -> Aircraft:
        """Return one non-Ownship aircraft by ID.

        Raises:
            KeyError: if `aircraft_id` is unknown, or is Ownship itself.
        """
        if aircraft_id == self.ownship.aircraft_id:
            raise KeyError(f"'{aircraft_id}' is Ownship, not a target.")
        return self.scenario.get_aircraft(aircraft_id)

    def alive_targets(self) -> list[Aircraft]:
        """Return every non-Ownship aircraft whose current state is alive."""
        return [
            aircraft
            for aircraft in self.get_targets()
            if self.scenario.get_state(aircraft.aircraft_id).alive
        ]

    def iff_capable_targets(self) -> list[Aircraft]:
        """Return every non-Ownship aircraft with a known IFF capability.

        No IFF logic exists yet, so every aircraft's `iff_capability`
        is still its "UNKNOWN" default; this method is scaffolding for
        a later phase and currently returns an empty list.
        """
        return [
            aircraft
            for aircraft in self.get_targets()
            if aircraft.iff_capability not in (None, "UNKNOWN")
        ]
