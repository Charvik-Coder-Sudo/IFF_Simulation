"""Immutable snapshot of one persistent logical IFF track.

Purpose:
    Defines `IFFTrack` — the persistent, cross-tick identity `IFFTrackManager`
    maintains for one aircraft — plus `TrackStatus`, `FriendFoeStatus`, and
    the deterministic `derive_friend_foe_status` rule they're built from.

Inputs:
    Built exclusively by `IFFTrackManager` from a `DecodedIFFMeasurement`
    (Phase 7) plus its own track-lifecycle bookkeeping.

Outputs:
    Consumed by `ReportGenerator` and any future consumer that needs
    "what does the interrogator currently believe about this aircraft."

Engineering explanation:
    Frozen (immutable) — `IFFTrack` is a point-in-time snapshot; the
    *mutable* bookkeeping (miss_count, consecutive-valid-reply streak)
    lives only inside `IFFTrackManager`'s private state, never on this
    public type. This mirrors every other per-instant record in this
    codebase (`RelativeState`, `SelectedTarget`, `ReplyMessage`,
    `DecodedIFFMeasurement`).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique

from ...domain import Vector3
from .authentication import AuthenticationResult
from .measurement import MeasurementStatus
from .mode import IFFMode
from .reply import ReplyType


@unique
class TrackStatus(Enum):
    """A track's position in the one-way lifecycle state machine.

    Engineering explanation:
        Strictly forward: None -> TENTATIVE -> CONFIRMED -> LOST. There
        is no demotion from CONFIRMED back to TENTATIVE — a confirmed
        track only ever ends by being lost.
    """

    TENTATIVE = "Tentative"
    CONFIRMED = "Confirmed"
    LOST = "Lost"


@unique
class FriendFoeStatus(Enum):
    """The track's derived Friend/Foe classification."""

    FRIENDLY = "Friendly"
    SUSPECT = "Suspect"
    UNKNOWN = "Unknown"


def derive_friend_foe_status(
    mode: IFFMode, authentication_result: bool, reply_status: MeasurementStatus
) -> FriendFoeStatus:
    """Deterministically classify Friend/Foe status from a measurement's own fields.

    Purpose:
        Implement exactly the four rules this phase specifies, as a
        pure, reusable, independently-testable function.

    Inputs:
        mode: the `IFFMode` the measurement answers.
        authentication_result: whether Mode 5 authentication passed.
        reply_status: `MeasurementStatus.VALID` or `NO_REPLY`.

    Outputs:
        One of `FriendFoeStatus.FRIENDLY`, `SUSPECT`, `UNKNOWN`.

    Mathematics:
        NO_REPLY                              -> UNKNOWN
        Mode 5 (L1 or L2) AND authenticated    -> FRIENDLY
        Mode 5 (L1 or L2) AND NOT authenticated -> SUSPECT
        Mode S                                 -> UNKNOWN

    Engineering reasoning:
        A pure function of already-decoded fields — no estimation, no
        state, no randomness. Mode S is always UNKNOWN here because it
        has no authentication mechanism in this simulator (see Phase 6's
        `ModeSReplyGenerator`), so it can never affirmatively establish
        friend or foe on its own.
    """
    if reply_status == MeasurementStatus.NO_REPLY:
        return FriendFoeStatus.UNKNOWN
    if mode in (IFFMode.MODE5_L1, IFFMode.MODE5_L2):
        return FriendFoeStatus.FRIENDLY if authentication_result else FriendFoeStatus.SUSPECT
    return FriendFoeStatus.UNKNOWN


@dataclass(frozen=True, slots=True)
class IFFTrack:
    """One persistent logical IFF track, as of its most recent update.

    Purpose:
        Carry everything downstream consumers need about one aircraft's
        ongoing IFF track: identity, latest geometry, mode/authentication
        status, and lifecycle bookkeeping (status/quality/miss history).

    Inputs:
        Constructed exclusively by `IFFTrackManager`; not intended to be
        hand-built by callers.

    Outputs:
        Consumed by `ReportGenerator.generate` and test/inspection code.

    Engineering explanation:
        `relative_velocity` and `reply_type` are `None` unless the
        caller supplies them to `IFFTrackManager.update` — Phase 7's
        frozen `DecodedIFFMeasurement` does not carry either field, and
        this phase must not modify that frozen type. A caller that
        still has the originating `SelectedTarget`/`ReplyMessage` on
        hand (e.g. the full pipeline driver script) can pass them
        through; a caller with only the `DecodedIFFMeasurement` gets
        `None` in those two fields rather than an invented value.
        `confidence` mirrors `track_quality` exactly: no separate
        confidence model is specified anywhere in this phase, and
        inventing a second, different formula would be undocumented
        behavior this phase's "no probabilistic scoring" constraint
        argues against.
    """

    track_id: int
    """Monotonically-assigned identifier for this track, unique for its
    lifetime (never reused, even after the track is lost)."""

    aircraft_id: str
    """The tracked aircraft's Scenario aircraft_id — the sole
    association key (see `IFFTrackManager`)."""

    ownship_id: str
    """The interrogating Ownship's aircraft_id."""

    time: float
    """The time of the measurement this snapshot reflects (updated on
    every tick, hit or miss)."""

    range_m: float
    """Slant range, meters — frozen at its last valid-reply value during
    a miss (a NO_REPLY carries no new geometry to update from)."""

    azimuth_deg: float
    """Azimuth, degrees — same freeze-on-miss behavior as `range_m`."""

    elevation_deg: float
    """Elevation, degrees — same freeze-on-miss behavior as `range_m`."""

    relative_velocity: Vector3 | None
    """Relative velocity, if supplied by the caller; otherwise None
    (see engineering explanation above)."""

    mode: IFFMode
    """The mode of the most recent valid reply (frozen during a miss)."""

    reply_status: MeasurementStatus
    """VALID or NO_REPLY for *this* tick's measurement specifically
    (unlike `track_status`, which reflects the track's overall lifecycle)."""

    mode_s_address: str | None
    """The Mode S logical ICAO address, if the track's mode is Mode S;
    None otherwise (frozen at its last valid-reply value during a miss)."""

    authentication_result: bool
    """Whether the most recent valid reply authenticated (frozen during
    a miss)."""

    friend_foe_status: FriendFoeStatus
    """Derived via `derive_friend_foe_status`, recomputed every tick
    (including misses, which always yield UNKNOWN)."""

    track_status: TrackStatus
    """This track's current lifecycle state."""

    track_quality: float
    """Deterministic quality score in [0, 1] (see `IFFTrackManager`'s
    quality state machine)."""

    last_update_time: float
    """The time of the most recent *valid* reply — unlike `time`, this
    does not advance during a miss."""

    sequence_number: int
    """The most recent measurement's sequence number (hit or miss)."""

    reply_type: ReplyType | None
    """The most recent valid reply's logical reply type, if supplied by
    the caller; otherwise None (see engineering explanation above)."""

    confidence: float
    """Mirrors `track_quality` exactly (see engineering explanation above)."""

    signal_strength: float
    """The most recent valid reply's (fixed, logical-only) signal
    strength; frozen during a miss."""

    propagation_delay: float | None
    """The most recent valid reply's propagation delay, microseconds;
    frozen during a miss. None if the track has never had a valid reply
    with propagation data available."""

    closing_velocity_mps: float | None = None
    """Phase 8.5 Part 1: rate at which range is shrinking, meters/second
    — carried through from the source `DecodedIFFMeasurement` (itself
    from `SelectedTarget`/`RelativeState`, never recomputed). Frozen
    during a miss, like `range_m`. Defaults to None for backward
    compatibility with pre-8.5 direct `IFFTrack(...)` constructions."""

    authentication_status: AuthenticationResult = AuthenticationResult.NOT_APPLICABLE
    """Phase 8.5 Part 2: the semantic authentication outcome (see
    `AuthenticationResult`), purely additive alongside the existing
    boolean `authentication_result` field."""

    def to_csv_row(self) -> dict:
        """Return this track as a dict, for Phase 8.5 Part 7's tracks.csv logging."""
        velocity = self.relative_velocity
        relative_velocity_csv = "" if velocity is None else f"{velocity.x};{velocity.y};{velocity.z}"
        return {
            "Time": self.time,
            "Track_ID": self.track_id,
            "Aircraft_ID": self.aircraft_id,
            "Ownship_ID": self.ownship_id,
            "Range": self.range_m,
            "Azimuth": self.azimuth_deg,
            "Elevation": self.elevation_deg,
            "Closing_Velocity": self.closing_velocity_mps,
            "Relative_Velocity": relative_velocity_csv,
            "Mode": self.mode.value,
            "Reply_Status": self.reply_status.value,
            "ModeS_Address": self.mode_s_address or "",
            "Authentication_Result": self.authentication_result,
            "Authentication_Status": self.authentication_status.value,
            "Friend_Foe_Status": self.friend_foe_status.value,
            "Track_Status": self.track_status.value,
            "Track_Quality": self.track_quality,
            "Confidence": self.confidence,
            "Last_Update_Time": self.last_update_time,
            "Sequence_Number": self.sequence_number,
            "Reply_Type": self.reply_type.value if self.reply_type is not None else "",
            "Signal_Strength": self.signal_strength,
            "Propagation_Delay": self.propagation_delay,
        }
