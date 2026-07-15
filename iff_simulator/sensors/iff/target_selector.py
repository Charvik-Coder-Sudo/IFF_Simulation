"""Determines which aircraft are visible to Ownship's IFF interrogator.

Purpose:
    Implements `TargetSelector`, the first stage of the IFF
    interrogator: for the current simulation tick, decide which
    aircraft (other than Ownship) can actually be interrogated. No
    interrogation pulses, replies, or tracking happen here — selection
    only.

Inputs:
    A `World` (already-completed Phase 2 runtime model), a
    `SelectionPolicy` (Strategy pattern, injected — defaults to
    `DefaultSelectionPolicy`), and a `GeometryEngine` reference
    (dependency-injected — defaults to the real `GeometryEngine`,
    which is stateless so the class itself is a valid "instance" to
    inject).

Outputs:
    `list[SelectedTarget]`, sorted by (Range, Aircraft ID) for
    deterministic ordering.

Engineering explanation:
    `TargetSelector` never computes geometry itself: every
    Range/Azimuth/Elevation/Bearing/Closing-Velocity value flows
    through exactly one `GeometryEngine.compute_batch` or
    `compute_relative_state` call. This keeps `GeometryEngine` the
    single source of truth and keeps `TargetSelector` a thin
    orchestration layer: gather candidates, ask GeometryEngine for
    their geometry, ask the policy which pass, sort, return.

    `World` is imported only under `TYPE_CHECKING`, not at runtime:
    `iff_simulator.simulation.world` already imports `Ownship` from
    this package (`sensors.iff`), so a real, eager `from ...simulation
    import World` here would create a circular import between the two
    packages. `TargetSelector` only ever uses `world` via duck-typed
    method calls (`world.ownship`, `world.get_targets()`, ...), so it
    never needs the real class at runtime — only type checkers do.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...geometry import GeometryEngine
from .selected_target import SelectedTarget
from .selection_policy import DefaultSelectionPolicy, SelectionPolicy

if TYPE_CHECKING:
    from ...simulation import World


class TargetSelector:
    """Selects which targets are visible to Ownship's IFF interrogator this tick.

    Purpose:
        The single entry point for "which aircraft can Ownship's IFF
        interrogator see right now."

    Inputs:
        world: the `World` holding Ownship, Scenario, and the clock.
        policy: a `SelectionPolicy` (Strategy pattern); defaults to
            `DefaultSelectionPolicy()` if not given.
        geometry_engine: the `GeometryEngine` to call into (dependency
            injection — a test can substitute a fake with the same
            `compute_batch`/`compute_relative_state` interface);
            defaults to the real `GeometryEngine`.

    Outputs:
        `select_targets()`, `select_one(target_id)`, `visible_targets()`.

    Engineering explanation:
        Runs in O(N) for N non-Ownship aircraft: one batched
        `GeometryEngine.compute_batch` call (numpy-vectorized, no
        per-target Python-level geometry math), one O(N) policy-filter
        pass, then the O(N log N) sort the spec requires for
        deterministic output ordering. No nested loops, no per-target
        search, no DataFrame anywhere in this class.
    """

    def __init__(
        self,
        world: World,
        policy: SelectionPolicy | None = None,
        geometry_engine: type[GeometryEngine] = GeometryEngine,
    ) -> None:
        self.world = world
        self.policy: SelectionPolicy = policy if policy is not None else DefaultSelectionPolicy()
        self.geometry_engine = geometry_engine

    def select_targets(self) -> list[SelectedTarget]:
        """Evaluate every non-Ownship aircraft and return the selected ones.

        Returns:
            `list[SelectedTarget]`, sorted by (range_m, target_id).
        """
        ownship = self.world.ownship
        candidates = [
            (aircraft, self.world.scenario.get_state(aircraft.aircraft_id))
            for aircraft in self.world.get_targets()
        ]
        if not candidates:
            return []

        relative_states = self.geometry_engine.compute_batch(
            self.world.current_time(),
            ownship.position,
            ownship.velocity,
            ownship.heading,
            [
                (aircraft.aircraft_id, aircraft_state.position, aircraft_state.velocity)
                for aircraft, aircraft_state in candidates
            ],
        )

        selected = [
            SelectedTarget.from_relative_state(relative_state)
            for (aircraft, aircraft_state), relative_state in zip(candidates, relative_states)
            if self.policy.accept(relative_state, aircraft, aircraft_state, ownship)
        ]
        selected.sort(key=lambda target: (target.range_m, target.target_id))
        return selected

    def select_one(self, target_id: str) -> SelectedTarget | None:
        """Evaluate a single aircraft by ID.

        Args:
            target_id: a non-Ownship Scenario aircraft_id.

        Returns:
            Its `SelectedTarget` if it passes the policy, else `None`.

        Raises:
            KeyError: if `target_id` is unknown, or is Ownship itself
                (per `World.get_target`).
        """
        ownship = self.world.ownship
        aircraft = self.world.get_target(target_id)
        aircraft_state = self.world.scenario.get_state(target_id)

        relative_state = self.geometry_engine.compute_relative_state(
            target_id,
            self.world.current_time(),
            ownship.position,
            ownship.velocity,
            ownship.heading,
            aircraft_state.position,
            aircraft_state.velocity,
        )

        if self.policy.accept(relative_state, aircraft, aircraft_state, ownship):
            return SelectedTarget.from_relative_state(relative_state)
        return None

    def visible_targets(self) -> list[str]:
        """Return just the target_ids of the currently selected targets.

        Returns:
            `list[str]`, in the same deterministic order as `select_targets()`.
        """
        return [target.target_id for target in self.select_targets()]
