"""InterrogationScheduler: decides when to transmit, to whom, and how.

Purpose:
    Implements `InterrogationScheduler`, the second stage of the IFF
    interrogator: given the targets `TargetSelector` says are visible
    this tick, decide whether it is time to transmit (per Ownship's
    `interrogation_rate`), which single target to interrogate (Strategy
    pattern, `SchedulingPolicy`), which mode to request (Strategy
    pattern, `ModeSelectionPolicy`), and build the resulting
    `InterrogationMessage`. No replies, decoding, transponders,
    tracking, or fusion happen here — selection and scheduling only.

Inputs:
    A `World` (for `current_time()` and `ownship`) and a `TargetSelector`
    (for `select_targets()`) — both dependency-injected, both reused
    from earlier phases without modification. Optionally a
    `SchedulingPolicy`, a `ModeSelectionPolicy`, and an
    `InterrogationQueue` (all dependency-injected; sensible defaults
    are used if omitted).

Outputs:
    `tick()` returns the `InterrogationMessage` transmitted this call,
    or `None` if nothing was transmitted (paused, not yet time, or no
    visible target). Every transmitted message is also enqueued onto
    the scheduler's `InterrogationQueue`.

Engineering explanation:
    `InterrogationScheduler` never performs geometry: every
    Range/Azimuth/Elevation value it uses came from `TargetSelector`
    (which itself got them from `GeometryEngine`). It also never
    creates its own clock — timing decisions are made entirely by
    comparing `World.current_time()` (the *same* `SimulationClock`
    `World` already owns) against an internally-tracked
    `next_transmission_time`, advanced by a fixed period each time a
    transmission slot is reached.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .interrogation import InterrogationMessage
from .interrogation_queue import InterrogationQueue
from .mode import DefaultModeSelectionPolicy, ModeSelectionPolicy
from .selected_target import SelectedTarget
from .uplink_format import DEFAULT_UPLINK_FORMAT_BY_MODE

if TYPE_CHECKING:
    # Imported only for type checking: `simulation.world` already imports
    # `Ownship` from this package (`sensors.iff`), so an eager, runtime
    # `from ...simulation import World` here would recreate the same
    # circular import `target_selector.py` avoids the same way (see its
    # module docstring). InterrogationScheduler only ever calls `world`
    # and `target_selector` via duck-typed method calls, so it never
    # needs the real classes at runtime.
    from ...simulation import World
    from .target_selector import TargetSelector


class SchedulingPolicy(ABC):
    """Abstract strategy: which single target (if any) to interrogate this instant.

    Purpose:
        Let `InterrogationScheduler` depend on an interface rather than
        a fixed priority rule, so later scheduling policies (priority,
        track quality, threat level, radar cueing, mission rules) can
        be injected without modifying the scheduler.

    Inputs:
        selected_targets: every currently visible target, as returned
            by `TargetSelector.select_targets()`.

    Outputs:
        The single `SelectedTarget` to interrogate this transmission
        instant, or `None` if there is nothing to interrogate.
    """

    @abstractmethod
    def choose(self, selected_targets: list[SelectedTarget]) -> SelectedTarget | None:
        """Choose one target (or none) to interrogate this transmission instant.

        Purpose:
            The single decision point every scheduling rule funnels
            through — "only one interrogation per transmission
            instant" is enforced by this method's return type (exactly
            zero or one target), not by the caller.
        Inputs:
            selected_targets: every currently visible target.
        Outputs:
            One `SelectedTarget`, or `None` if `selected_targets` is empty.
        """
        raise NotImplementedError


class DefaultSchedulingPolicy(SchedulingPolicy):
    """Closest target first, ties broken by lowest Aircraft ID.

    Purpose:
        Implement the required default priority: closest target, then
        lowest Aircraft ID.

    Engineering explanation:
        `TargetSelector.select_targets()` is already guaranteed (Phase
        4's own deterministic-ordering contract) to be sorted by
        `(range_m, target_id)` — exactly this priority order — so this
        policy does not re-sort or re-derive priority itself; it simply
        takes the first element. Re-deriving the same ordering here
        would be exactly the kind of duplicated logic this phase must
        avoid.
    """

    def choose(self, selected_targets: list[SelectedTarget]) -> SelectedTarget | None:
        return selected_targets[0] if selected_targets else None


class InterrogationScheduler:
    """Decides when to transmit, to whom, and with what mode/format.

    Purpose:
        The single entry point for "should Ownship's IFF interrogator
        transmit right now, and if so, what interrogation message
        results."

    Inputs:
        world: the `World` holding Ownship and the (shared)
            `SimulationClock`.
        target_selector: the `TargetSelector` to ask for visible
            targets.
        scheduling_policy: a `SchedulingPolicy` (Strategy pattern);
            defaults to `DefaultSchedulingPolicy()`.
        mode_selection_policy: a `ModeSelectionPolicy` (Strategy
            pattern); defaults to `DefaultModeSelectionPolicy()`.
        queue: an `InterrogationQueue`; defaults to a new, empty one.

    Outputs:
        `tick()`, `pause()`, `resume()`, `is_paused()`, `period`.

    Engineering explanation:
        Runs in O(1) when not transmitting, O(N) when transmitting (one
        `TargetSelector.select_targets()` call, itself O(N log N) —
        see Phase 4's complexity notes). No nested loops, no per-target
        search, no DataFrame anywhere in this class.
    """

    def __init__(
        self,
        world: "World",
        target_selector: "TargetSelector",
        scheduling_policy: SchedulingPolicy | None = None,
        mode_selection_policy: ModeSelectionPolicy | None = None,
        queue: InterrogationQueue | None = None,
    ) -> None:
        self.world = world
        self.target_selector = target_selector
        self.scheduling_policy: SchedulingPolicy = scheduling_policy or DefaultSchedulingPolicy()
        self.mode_selection_policy: ModeSelectionPolicy = (
            mode_selection_policy or DefaultModeSelectionPolicy()
        )
        self.queue = queue if queue is not None else InterrogationQueue()

        self._next_sequence_number = 1
        self._next_transmission_time = world.current_time()
        self._paused = False

    @property
    def period(self) -> float:
        """Interrogation period, seconds: 1 / Ownship.interrogation_rate.

        Returns:
            `math.inf` if the configured rate is <= 0 (an interrogator
            with no configured rate is effectively off, and never
            transmits — this avoids a ZeroDivisionError rather than
            raising one).
        """
        rate = self.world.ownship.interrogation_rate
        if rate <= 0:
            return math.inf
        return 1.0 / rate

    def pause(self) -> None:
        """Stop transmitting until `resume()` is called. Does not reset timing:
        the next transmission after `resume()` still honors the schedule."""
        self._paused = True

    def resume(self) -> None:
        """Resume transmitting after `pause()`."""
        self._paused = False

    def is_paused(self) -> bool:
        """Return whether the scheduler is currently paused."""
        return self._paused

    def tick(self) -> InterrogationMessage | None:
        """Evaluate the current simulation tick and transmit if it is time to.

        Returns:
            The `InterrogationMessage` transmitted this call, or `None`
            if the scheduler is paused, it is not yet time to transmit,
            the interrogation rate is <= 0, or no target is currently
            selectable.

        Engineering reasoning:
            Even when no target is selectable, a due transmission slot
            still consumes its place in the schedule (advances
            `next_transmission_time` by one period): the interrogator's
            timing does not stall waiting for a target to appear. Only
            an *actual* transmission (a chosen target) consumes a
            sequence number and enqueues a message — an empty slot is
            not "transmitted" in the IFF sense.
        """
        if self._paused:
            return None

        current_time = self.world.current_time()
        if current_time < self._next_transmission_time:
            return None

        period = self.period
        if math.isinf(period):
            return None

        selected_targets = self.target_selector.select_targets()
        chosen = self.scheduling_policy.choose(selected_targets)
        self._next_transmission_time += period

        if chosen is None:
            return None

        mode = self.mode_selection_policy.select_mode(chosen)
        uplink_format = DEFAULT_UPLINK_FORMAT_BY_MODE[mode]

        message = InterrogationMessage.from_selected_target(
            chosen,
            sequence_number=self._next_sequence_number,
            ownship_id=self.world.ownship.aircraft_id,
            mode=mode,
            uplink_format=uplink_format,
        )
        self._next_sequence_number += 1
        self.queue.enqueue(message)
        return message
