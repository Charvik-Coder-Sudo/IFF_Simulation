"""Aircraft configuration/metadata domain object.

Purpose:
    Represent the static, non-kinematic identity of a single aircraft:
    who/what it is, not where it is right now. This is deliberately
    separate from `AircraftState`, which owns the live kinematic data.

Inputs:
    Values parsed by `GroundTruthLoader` from a `.tdf` file's own
    target header line (currently just an aircraft ID).

Outputs:
    An immutable `Aircraft` instance, stored in a `Scenario`.

Engineering explanation:
    IFF identity, Mode capability, and mode data are represented here as
    structural placeholder fields only. Phase 1 has no IFF, Mode S, or
    Mode 5 logic, so these fields hold neutral defaults
    ("UNKNOWN"/empty) and exist purely so later phases can populate them
    without changing this class's shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Aircraft:
    """Static configuration/metadata for one aircraft.

    Purpose:
        Hold everything about an aircraft that does not change from
        sample to sample: its identifier, callsign, IFF-related
        placeholders, and its motion model label.

    Inputs:
        aircraft_id: unique identifier (e.g. "TARGET_1").
        callsign: human-readable callsign, if known.
        identity: IFF identity placeholder (friend/foe/unknown);
            unpopulated in Phase 1.
        iff_capability: IFF Mode capability placeholder; unpopulated in
            Phase 1.
        mode_data: raw Mode 1/2/3/4/5/S data placeholder; empty in
            Phase 1.
        motion_model: label describing how this aircraft's position
            evolves; Phase 1 aircraft simply replay a recorded
            trajectory, so this defaults to "RECORDED_TRAJECTORY".

    Outputs:
        An immutable value object stored in `Scenario.get_aircraft()`.

    Engineering explanation:
        Frozen because aircraft identity/configuration should never be
        mutated at runtime — only its associated `AircraftState` changes
        over time.
    """

    aircraft_id: str
    callsign: str = ""
    identity: str = "UNKNOWN"
    iff_capability: str = "UNKNOWN"
    mode_data: dict = field(default_factory=dict)
    motion_model: str = "RECORDED_TRAJECTORY"
