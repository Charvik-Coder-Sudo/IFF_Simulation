"""Strategy pattern for deciding whether a target is selectable this tick.

Purpose:
    Defines `SelectionPolicy` (the abstract strategy) and
    `DefaultSelectionPolicy` (the four required Phase 4 rules: alive,
    IFF capable, within maximum range, within antenna beam), so
    `TargetSelector` never hard-codes selection rules itself and later
    phases can inject a different policy without touching it.

Inputs:
    A `RelativeState` (from `GeometryEngine`), the target's `Aircraft`
    (static metadata) and `AircraftState` (live kinematics), and the
    `Ownship` making the selection.

Outputs:
    `bool` — whether the target should be selected.

Engineering explanation:
    `accept()`'s signature deliberately includes `Aircraft` in addition
    to `RelativeState`/`AircraftState`/`Ownship`: the "IFF capable" rule
    needs `Aircraft.iff_capability`, which lives on the aircraft's
    static metadata, not its per-tick kinematic `AircraftState`. There
    is no way to evaluate that rule without it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ...domain import Aircraft, AircraftState
from ...geometry import RelativeState
from .ownship import Ownship

#: Aircraft.iff_capability values that mean "not actually IFF capable."
#: No boolean IFF-capability flag exists on `Aircraft` (Phase 1.5 never
#: assigns one — no IFF logic exists yet), so "capable" is defined the
#: same way `World.iff_capable_targets()` (Phase 2) already defines it:
#: any value other than the "UNKNOWN" default. No new field is added.
_NOT_IFF_CAPABLE = (None, "UNKNOWN")


class SelectionPolicy(ABC):
    """Abstract strategy: decide whether one target should be selected this tick.

    Purpose:
        Let `TargetSelector` depend on an interface instead of a fixed
        rule set (Strategy pattern / dependency injection), so future
        policies (priority, track quality, threat level, radar cueing,
        mission rules) can be added by writing a new `SelectionPolicy`
        subclass — never by editing `TargetSelector`.

    Inputs:
        See `accept`.

    Outputs:
        See `accept`.

    Engineering explanation:
        A pure interface: no state, no default behavior. All four
        Phase 4 rules live in `DefaultSelectionPolicy` below.
    """

    @abstractmethod
    def accept(
        self,
        relative_state: RelativeState,
        aircraft: Aircraft,
        aircraft_state: AircraftState,
        ownship: Ownship,
    ) -> bool:
        """Decide whether this target should be selected this tick.

        Purpose:
            The single decision point every selection rule funnels through.
        Inputs:
            relative_state: this target's `RelativeState`, computed by
                `GeometryEngine` (never recomputed here).
            aircraft: the target's static `Aircraft` metadata.
            aircraft_state: the target's live `AircraftState`.
            ownship: the `Ownship` performing the selection (for its
                maximum_range/beam_width/beam_height configuration).
        Outputs:
            True if the target should be selected, False otherwise.
        Units:
            N/A (boolean decision).
        Mathematics:
            Defined by the concrete policy.
        Engineering reasoning:
            Kept as a single boolean predicate (not e.g. a list of
            failed-rule reasons) since Phase 4 is selection-only; a
            later phase can extend this return type if reasons become
            necessary, without changing the Strategy pattern itself.
        """
        raise NotImplementedError


class DefaultSelectionPolicy(SelectionPolicy):
    """The four required Phase 4 selection rules.

    Purpose:
        Implement exactly: (1) alive, (2) IFF capable, (3) within
        Ownship's maximum range, (4) within Ownship's antenna beam
        (azimuth and elevation). Nothing else.

    Inputs:
        See `SelectionPolicy.accept`.

    Outputs:
        See `SelectionPolicy.accept`.

    Engineering explanation:
        Extension points for later filters (priority, track quality,
        threat level, radar cueing, mission rules) are deliberately
        *not* implemented here — a later phase adds them as additional
        `SelectionPolicy` implementations (or a composite policy that
        chains several), never by modifying this class's four rules.
    """

    def accept(
        self,
        relative_state: RelativeState,
        aircraft: Aircraft,
        aircraft_state: AircraftState,
        ownship: Ownship,
    ) -> bool:
        return (
            self._is_alive(aircraft_state)
            and self._is_iff_capable(aircraft)
            and self._within_maximum_range(relative_state, ownship)
            and self._within_antenna_beam(relative_state, ownship)
        )

    @staticmethod
    def _is_alive(aircraft_state: AircraftState) -> bool:
        """Rule 1: Alive == True."""
        return aircraft_state.alive

    @staticmethod
    def _is_iff_capable(aircraft: Aircraft) -> bool:
        """Rule 2: IFF_Capable == True (see module docstring for the
        "UNKNOWN" default -> not-capable mapping this uses)."""
        return aircraft.iff_capability not in _NOT_IFF_CAPABLE

    @staticmethod
    def _within_maximum_range(relative_state: RelativeState, ownship: Ownship) -> bool:
        """Rule 3: Range <= Ownship.maximum_range (inclusive)."""
        return relative_state.range_m <= ownship.maximum_range

    @staticmethod
    def _within_antenna_beam(relative_state: RelativeState, ownship: Ownship) -> bool:
        """Rule 4: abs(azimuth) <= beam_width/2 and abs(elevation) <= beam_height/2
        (both inclusive).

        Engineering note:
            Per the Phase 4 spec, this uses `RelativeState`'s own
            Azimuth/Elevation (GeometryEngine's boresight/math-frame
            angles) directly, exactly as computed — not a
            heading-relative angle. A real gimbaled or body-fixed
            antenna would need the beam check expressed relative to
            Ownship's heading (closer to `RelativeState`'s Bearing,
            recentered to a signed angle); that transformation is new
            geometry this phase must not introduce, so it is left as a
            documented extension point for a future `SelectionPolicy`
            rather than implemented here.
        """
        return (
            abs(relative_state.azimuth_deg) <= ownship.beam_width / 2.0
            and abs(relative_state.elevation_deg) <= ownship.beam_height / 2.0
        )
