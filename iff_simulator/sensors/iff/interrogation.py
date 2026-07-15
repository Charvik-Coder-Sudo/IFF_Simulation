"""Immutable record of one transmitted (or about-to-be-transmitted) interrogation.

Purpose:
    Defines `InterrogationMessage`, the logical interrogation the
    scheduler transmits to one target: who sent it, who it's for, what
    mode/format was requested, and the target's geometry at that
    instant.

Inputs:
    Built exclusively from a `SelectedTarget` (Phase 4) plus the
    scheduling metadata (sequence number, Ownship ID, mode, uplink
    format) the scheduler assigns; never computes geometry itself.

Outputs:
    Consumed by `InterrogationQueue` (enqueued for a future receiver
    stage — not implemented yet) and by `interrogations.csv` export.

Engineering explanation:
    Frozen (immutable), for the same reason `SelectedTarget` and
    `RelativeState` are: a transmitted interrogation is a fact about
    one instant that must never be mutated after creation.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...domain import Vector3
from .mode import IFFMode
from .selected_target import SelectedTarget
from .uplink_format import UplinkFormat


@dataclass(frozen=True, slots=True)
class InterrogationMessage:
    """One logical IFF interrogation.

    Purpose:
        Carry everything a downstream stage (a future receiver/reply
        model) needs to know about one interrogation: timing, identity,
        requested mode/format, and target geometry.

    Inputs:
        Constructed via `InterrogationMessage.from_selected_target`;
        not intended to be hand-built by callers.

    Outputs:
        Consumed by `InterrogationQueue` and CSV export.

    Engineering explanation:
        `range_m`/`azimuth_deg`/`elevation_deg` follow this codebase's
        existing unit-suffix convention (matching `SelectedTarget`,
        `RelativeState`, `AircraftState`) rather than the bare names
        ("range", "azimuth", ...) used in the spec's prose — the CSV
        export (`interrogations.csv`) uses the spec's exact column
        names ("Range", "Azimuth", ...) at the file-format boundary,
        keeping the Python attribute names internally consistent.
    """

    time: float
    """Simulation time this interrogation was transmitted at (copied
    verbatim from the source SelectedTarget — never recomputed)."""

    sequence_number: int
    """Strictly monotonic, never-reused transmission sequence number,
    starting at 1."""

    ownship_id: str
    """Aircraft ID of the Ownship that transmitted this interrogation."""

    target_id: str
    """Aircraft ID of the interrogated target."""

    mode: IFFMode
    """The IFF mode requested for this interrogation."""

    uplink_format: UplinkFormat
    """The logical uplink format label for this interrogation."""

    range_m: float
    """Slant range to the target, meters (from SelectedTarget)."""

    azimuth_deg: float
    """Azimuth to the target, degrees (from SelectedTarget)."""

    elevation_deg: float
    """Elevation to the target, degrees (from SelectedTarget)."""

    closing_velocity_mps: float = 0.0
    """Phase 8.5 Part 1: rate at which range is shrinking, meters/second;
    positive = approaching (from SelectedTarget/RelativeState). Defaults
    to 0.0 so pre-8.5 direct `InterrogationMessage(...)` constructions
    (throughout the existing test suite) keep working unmodified."""

    relative_velocity: Vector3 | None = None
    """Phase 8.5 Part 1: target velocity minus Ownship velocity,
    meters/second, ENU (from SelectedTarget/RelativeState). Carrying
    this through the pipeline means `IFFTrackManager` (and anything
    else downstream) never needs a second `GeometryEngine` call to
    recover it — see this class's `from_selected_target`. Defaults to
    None for the same backward-compatibility reason as
    `closing_velocity_mps`."""

    @classmethod
    def from_selected_target(
        cls,
        selected_target: SelectedTarget,
        sequence_number: int,
        ownship_id: str,
        mode: IFFMode,
        uplink_format: UplinkFormat,
    ) -> "InterrogationMessage":
        """Build an InterrogationMessage from a SelectedTarget plus scheduling metadata.

        Purpose:
            The single place a `SelectedTarget` is turned into a
            transmittable interrogation — avoids repeating this field
            mapping at every call site.
        Inputs:
            selected_target: the target being interrogated this tick.
            sequence_number: this transmission's sequence number.
            ownship_id: the transmitting Ownship's aircraft ID.
            mode: the `IFFMode` chosen by the mode-selection policy.
            uplink_format: the `UplinkFormat` for that mode.
        Outputs:
            A new `InterrogationMessage`.
        Engineering reasoning:
            Pure field copy for time/range/azimuth/elevation/closing
            velocity/relative velocity — geometry and timing come
            directly from `SelectedTarget` (itself sourced from the
            single `GeometryEngine.compute_batch` call
            `TargetSelector.select_targets()` already made this tick),
            never recomputed here. Carrying `closing_velocity_mps` and
            `relative_velocity` forward (Phase 8.5 Part 1) is what lets
            `IFFTrackManager` receive full relative geometry without a
            second geometry computation.
        """
        return cls(
            time=selected_target.time,
            sequence_number=sequence_number,
            ownship_id=ownship_id,
            target_id=selected_target.target_id,
            mode=mode,
            uplink_format=uplink_format,
            range_m=selected_target.range_m,
            azimuth_deg=selected_target.azimuth_deg,
            elevation_deg=selected_target.elevation_deg,
            closing_velocity_mps=selected_target.closing_velocity_mps,
            relative_velocity=selected_target.relative_velocity,
        )

    def to_csv_row(self) -> dict:
        """Return this message as a dict keyed by the interrogations.csv column names.

        Purpose:
            The single place the Python attribute names are mapped to
            the spec's exact CSV column names.
        Outputs:
            dict with keys: Time, Sequence, Ownship_ID, Target_ID, Mode,
            UF, Range, Azimuth, Elevation, Closing_Velocity,
            Relative_Velocity (the last two added by Phase 8.5 Part 1 —
            appended after the original columns, so any consumer reading
            only the first 9 columns is unaffected).
        """
        velocity = self.relative_velocity
        relative_velocity_csv = "" if velocity is None else f"{velocity.x};{velocity.y};{velocity.z}"
        return {
            "Time": self.time,
            "Sequence": self.sequence_number,
            "Ownship_ID": self.ownship_id,
            "Target_ID": self.target_id,
            "Mode": self.mode.value,
            "UF": self.uplink_format.value,
            "Range": self.range_m,
            "Azimuth": self.azimuth_deg,
            "Elevation": self.elevation_deg,
            "Closing_Velocity": self.closing_velocity_mps,
            "Relative_Velocity": relative_velocity_csv,
        }
