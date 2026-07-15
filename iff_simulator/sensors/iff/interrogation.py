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
            Pure field copy for time/range/azimuth/elevation — geometry
            and timing come directly from `SelectedTarget` (itself
            sourced from `GeometryEngine`), never recomputed here.
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
        )

    def to_csv_row(self) -> dict:
        """Return this message as a dict keyed by the interrogations.csv column names.

        Purpose:
            The single place the Python attribute names are mapped to
            the spec's exact CSV column names.
        Outputs:
            dict with keys: Time, Sequence, Ownship_ID, Target_ID, Mode,
            UF, Range, Azimuth, Elevation.
        """
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
        }
