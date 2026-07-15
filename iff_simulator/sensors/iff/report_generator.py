"""Turns an IFFTrack into the exact-schema IFFMeasurementReport.

Purpose:
    Defines `IFFMeasurementReport` (a deliberately trimmed view of
    `IFFTrack` — exactly the fields this phase specifies, no more) and
    `ReportGenerator`, which builds one from an `IFFTrack`.

Inputs:
    An `IFFTrack`.

Outputs:
    An `IFFMeasurementReport`.

Engineering explanation:
    `IFFMeasurementReport` intentionally omits three `IFFTrack` fields:
    `last_update_time`, `reply_type`, and `confidence` — this phase's
    "the report must contain exactly [...] no extra fields" is explicit
    about the field list, and those three are not in it. `confidence`
    in particular is omitted because it is a pure mirror of
    `track_quality` (see `track.py`); the report carries `track_quality`
    only, avoiding a redundant duplicate column.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ...domain import Vector3
from .measurement import MeasurementStatus
from .mode import IFFMode
from .track import FriendFoeStatus, IFFTrack, TrackStatus


@dataclass(frozen=True, slots=True)
class IFFMeasurementReport:
    """The exact-schema measurement report derived from one IFFTrack.

    Purpose:
        Carry precisely the fields this phase's Part 7 lists — the
        reporting-layer view of a track, distinct from the track's own
        (richer) internal representation.

    Inputs:
        Constructed exclusively by `ReportGenerator.generate`.

    Outputs:
        Consumed by `ReportWriter`.
    """

    time: float
    track_id: int
    aircraft_id: str
    ownship_id: str
    range_m: float
    azimuth_deg: float
    elevation_deg: float
    relative_velocity: Vector3 | None
    mode: IFFMode
    reply_status: MeasurementStatus
    mode_s_address: str | None
    authentication_result: bool
    friend_foe_status: FriendFoeStatus
    track_quality: float
    track_status: TrackStatus
    sequence_number: int
    signal_strength: float
    propagation_delay: float | None


class ReportGenerator:
    """Builds an IFFMeasurementReport from an IFFTrack.

    Purpose:
        The single place an `IFFTrack` is narrowed down to the report
        schema — avoids repeating this field list at every call site.

    Inputs:
        `generate(track)` / `generate_many(tracks)`.

    Outputs:
        `IFFMeasurementReport` / `list[IFFMeasurementReport]`.

    Engineering explanation:
        Pure field copy, no recomputation — every value already exists
        on the `IFFTrack`; `ReportGenerator` never estimates or derives
        anything new.
    """

    def generate(self, track: IFFTrack) -> IFFMeasurementReport:
        """Build one IFFMeasurementReport from one IFFTrack."""
        return IFFMeasurementReport(
            time=track.time,
            track_id=track.track_id,
            aircraft_id=track.aircraft_id,
            ownship_id=track.ownship_id,
            range_m=track.range_m,
            azimuth_deg=track.azimuth_deg,
            elevation_deg=track.elevation_deg,
            relative_velocity=track.relative_velocity,
            mode=track.mode,
            reply_status=track.reply_status,
            mode_s_address=track.mode_s_address,
            authentication_result=track.authentication_result,
            friend_foe_status=track.friend_foe_status,
            track_quality=track.track_quality,
            track_status=track.track_status,
            sequence_number=track.sequence_number,
            signal_strength=track.signal_strength,
            propagation_delay=track.propagation_delay,
        )

    def generate_many(self, tracks: Iterable[IFFTrack]) -> list[IFFMeasurementReport]:
        """Build one IFFMeasurementReport per IFFTrack, preserving input order."""
        return [self.generate(track) for track in tracks]
