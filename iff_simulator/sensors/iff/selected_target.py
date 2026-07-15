"""Immutable record of one target selected as visible to Ownship's IFF interrogator.

Purpose:
    Defines `SelectedTarget`, the output type of `TargetSelector`: the
    geometric facts about one target that passed every selection rule
    (alive, IFF capable, within max range, within antenna beam) this
    tick. This is a *decision* record (this target IS selectable), not
    a raw geometry fact — that distinction is why it is its own type
    rather than reusing `RelativeState` directly, even though its
    fields are drawn straight from one.

Inputs:
    Built exclusively from a `RelativeState` already computed by
    `GeometryEngine`; never computes anything itself.

Outputs:
    Consumed by whatever later IFF stage decides what to do with a
    visible target (interrogation scheduling, reply handling — neither
    exists yet).

Engineering explanation:
    Frozen (immutable), for the same reason `RelativeState` is: this
    is a computed fact about one instant and must never be mutated
    after creation.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...domain import Vector3
from ...geometry import RelativeState


@dataclass(frozen=True, slots=True)
class SelectedTarget:
    """A target that passed IFF target-selection this tick.

    Purpose:
        Carry exactly the fields a downstream IFF stage needs about a
        selected target: identity, time, range/azimuth/elevation,
        closing velocity, and relative kinematics.

    Inputs:
        Constructed via `SelectedTarget.from_relative_state`; not
        intended to be hand-built by callers.

    Outputs:
        Consumed by later IFF stages (interrogation scheduling, etc. —
        not implemented in this phase).

    Engineering explanation:
        Field names follow this codebase's existing unit-suffix
        convention (`range_m`, `azimuth_deg`, `elevation_deg`,
        `closing_velocity_mps`), matching `RelativeState` and
        `AircraftState` exactly, rather than the bare names ("range",
        "azimuth", ...) used in the spec's prose — consistent naming
        across the domain model outweighs matching the prose literally.
        `bearing_deg` (heading-relative) is intentionally not carried
        over: target selection reasons about `RelativeState`'s
        boresight-relative Azimuth/Elevation, not the navigation-style
        Bearing.
    """

    time: float
    """Simulation time this selection was evaluated at."""

    target_id: str
    """Identifier of the selected target (matches its Scenario aircraft_id)."""

    range_m: float
    """Slant range from Ownship to the target, meters. Always >= 0."""

    azimuth_deg: float
    """Boresight-relative azimuth to the target, degrees (GeometryEngine convention)."""

    elevation_deg: float
    """Boresight-relative elevation to the target, degrees (GeometryEngine convention)."""

    closing_velocity_mps: float
    """Rate at which range is shrinking, meters/second; positive = approaching."""

    relative_position: Vector3
    """Target position minus Ownship position, meters, ENU."""

    relative_velocity: Vector3
    """Target velocity minus Ownship velocity, meters/second, ENU."""

    @classmethod
    def from_relative_state(cls, relative_state: RelativeState) -> "SelectedTarget":
        """Build a SelectedTarget by copying fields verbatim from a RelativeState.

        Purpose:
            The single place a `RelativeState` is narrowed down to the
            fields target selection cares about — avoids repeating this
            field list at every call site.
        Inputs:
            relative_state: a `RelativeState` already computed by
                `GeometryEngine` for this target/time.
        Outputs:
            A new `SelectedTarget`.
        Engineering reasoning:
            Pure field copy, no recomputation — `GeometryEngine` remains
            the single source of truth for every one of these values.
        """
        return cls(
            time=relative_state.time,
            target_id=relative_state.target_id,
            range_m=relative_state.range_m,
            azimuth_deg=relative_state.azimuth_deg,
            elevation_deg=relative_state.elevation_deg,
            closing_velocity_mps=relative_state.closing_velocity_mps,
            relative_position=relative_state.relative_position,
            relative_velocity=relative_state.relative_velocity,
        )
