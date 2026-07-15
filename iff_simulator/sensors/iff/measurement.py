"""The final decoded IFF measurement produced by the receive pipeline.

Purpose:
    Defines `DecodedIFFMeasurement` — the unified, cross-mode output of
    the entire receive chain (Propagation -> Receiver -> ReplyMatcher ->
    ModeDecoder) — and `MeasurementStatus`, its top-level VALID/NO_REPLY
    outcome.

Inputs:
    Built exclusively by `ModeDecoder.decode`.

Outputs:
    Consumed by any future stage that needs "what did the interrogator
    learn about this target this tick" without caring which mode
    answered (a future tracking/fusion phase — not implemented here).

Engineering explanation:
    Deliberately a *lean*, cross-mode-common schema, not a full re-export
    of every Mode S/Mode 5 payload field: `icao_address`/`mission` are
    simply `None` for modes that don't have that concept (Mode 5/Mode S
    respectively), and payload details this schema doesn't carry
    (altitude, capability, platform_id/platform_address) remain
    available on the original `ReplyMessage.payload` for a caller that
    needs them — `DecodedIFFMeasurement` is the "was this target
    identified, and how" summary a fusion stage would actually consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique

from ...domain import Vector3
from .authentication import AuthenticationResult
from .mode import IFFMode


@unique
class MeasurementStatus(Enum):
    """Top-level outcome of one decode attempt.

    Purpose:
        Give every interrogation exactly one of two possible outcomes,
        per this phase's validation requirement: a reply arrived and
        was decoded (`VALID` — regardless of whether Mode 5
        authentication within it passed; that detail lives in
        `DecodedIFFMeasurement.authentication_result`), or it did not
        (`NO_REPLY`).

    Engineering explanation:
        Deliberately distinct from `ReplyMessage`'s own `ReplyStatus`
        (OK/FAILED_AUTH), which describes a reply that *did* arrive.
        `MeasurementStatus.NO_REPLY` only occurs when no reply arrived
        at all (transponder never responded, identity mismatch, or
        timeout) — a `FAILED_AUTH` reply is still `MeasurementStatus.VALID`.

        Phase 8.5 Part 3 extends this enum with placeholders for
        outcomes a future RF-realism phase might distinguish
        (`TIMEOUT`, `GARBLED`, `FRUITED`, `LATE_REPLY`, `CRC_ERROR`,
        `UNKNOWN_MODE`). Only `VALID` and `NO_REPLY` are ever assigned
        by `ModeDecoder` today; the rest are structural placeholders,
        exactly like `ReplyMessage`'s legacy mode1/2/3A/C fields —
        present for schema stability, not yet implemented behavior.
    """

    VALID = "VALID"
    NO_REPLY = "NO_REPLY"

    # --- Phase 8.5 Part 3: placeholders for future phases only ---
    TIMEOUT = "TIMEOUT"
    GARBLED = "GARBLED"
    FRUITED = "FRUITED"
    LATE_REPLY = "LATE_REPLY"
    CRC_ERROR = "CRC_ERROR"
    UNKNOWN_MODE = "UNKNOWN_MODE"


@dataclass(frozen=True, slots=True)
class DecodedIFFMeasurement:
    """The final, cross-mode decoded result of one interrogation.

    Purpose:
        Carry everything a downstream consumer needs about one
        interrogation's outcome: identity/geometry/timing, in one
        uniform shape regardless of which mode (or no mode) answered.

    Inputs:
        Constructed exclusively by `ModeDecoder.decode`.

    Outputs:
        Consumed by any future tracking/fusion stage (not implemented
        in this phase).

    Engineering explanation:
        `measurement_id` and `sequence_number` are both set to the
        source interrogation's own `sequence_number` — the same
        "reuse the interrogation's identity, never invent a counter"
        pattern `ReplyMessage.reply_id` already established in Phase 6,
        so every measurement's identity is reproducible purely from
        its (Ground Truth, Interrogation, Reply) inputs.
    """

    measurement_id: int
    """Equal to the source interrogation's sequence_number (see
    engineering explanation above)."""

    time: float
    """The source interrogation's time, copied verbatim."""

    target_id: str
    """The interrogated aircraft's ID, copied verbatim."""

    ownship_id: str
    """The interrogating Ownship's ID, copied verbatim."""

    mode: IFFMode
    """The IFF mode this measurement answers, copied verbatim."""

    range_m: float
    """Slant range, meters — copied verbatim from the interrogation
    (itself sourced from GeometryEngine via Phase 4); never recomputed."""

    azimuth_deg: float
    """Azimuth, degrees — copied verbatim from the interrogation."""

    elevation_deg: float
    """Elevation, degrees — copied verbatim from the interrogation."""

    icao_address: str | None
    """The Mode S logical ICAO address, or None (Mode 5 replies, or
    MeasurementStatus.NO_REPLY, have no ICAO address)."""

    authentication_result: bool
    """Whether the reply authenticated (always False for Mode S, and
    for NO_REPLY)."""

    identity: str
    """One of BLUE/RED/NEUTRAL/UNKNOWN. "UNKNOWN" for NO_REPLY."""

    mission: str | None
    """The reported mission type (Mode 5 only), or None (Mode S, or
    NO_REPLY, has no mission concept)."""

    reply_status: MeasurementStatus
    """VALID (a reply was decoded) or NO_REPLY."""

    processing_delay: float | None
    """The transponder's processing delay, microseconds, copied
    verbatim from the reply; None for NO_REPLY."""

    propagation_delay: float | None
    """The reply's propagation delay, microseconds, from
    `PropagatedReply`; None for NO_REPLY."""

    arrival_time: float | None
    """The reply's computed arrival time, from `PropagatedReply`; None
    for NO_REPLY."""

    sequence_number: int
    """Equal to the source interrogation's sequence_number (see
    `measurement_id`)."""

    closing_velocity_mps: float | None = None
    """Phase 8.5 Part 1: copied verbatim from the source interrogation
    (itself carried through from `SelectedTarget`/`RelativeState` —
    never recomputed). Defaults to None so pre-8.5 direct
    `DecodedIFFMeasurement(...)` constructions keep working unmodified."""

    relative_velocity: Vector3 | None = None
    """Phase 8.5 Part 1: copied verbatim from the source interrogation.
    Defaults to None for the same backward-compatibility reason as
    `closing_velocity_mps`."""

    authentication_status: AuthenticationResult = AuthenticationResult.NOT_APPLICABLE
    """Phase 8.5 Part 2: the semantic authentication outcome (see
    `AuthenticationResult`), derived from `authentication_result` and
    `mode`. Purely additive alongside the existing boolean
    `authentication_result` field — see that class's docstring for why
    the boolean was not replaced."""

    signal_strength: float | None = None
    """Phase 8.5 Part 4: copied verbatim from the matched
    `PropagatedReply.signal_strength` (see `propagation.py`'s
    deterministic inverse-square model); None for NO_REPLY. Lets
    `IFFTrackManager` track signal strength history without a second
    propagation computation."""

    def to_csv_row(self) -> dict:
        """Return this measurement as a dict, for Phase 8.5 Part 7's decoded.csv logging."""
        velocity = self.relative_velocity
        relative_velocity_csv = "" if velocity is None else f"{velocity.x};{velocity.y};{velocity.z}"
        return {
            "Time": self.time,
            "Sequence": self.sequence_number,
            "Target_ID": self.target_id,
            "Ownship_ID": self.ownship_id,
            "Mode": self.mode.value,
            "Range": self.range_m,
            "Azimuth": self.azimuth_deg,
            "Elevation": self.elevation_deg,
            "Closing_Velocity": self.closing_velocity_mps,
            "Relative_Velocity": relative_velocity_csv,
            "ICAO_Address": self.icao_address or "",
            "Authentication_Result": self.authentication_result,
            "Authentication_Status": self.authentication_status.value,
            "Identity": self.identity,
            "Mission": self.mission or "",
            "Reply_Status": self.reply_status.value,
            "Processing_Delay": self.processing_delay,
            "Propagation_Delay": self.propagation_delay,
            "Signal_Strength": self.signal_strength,
            "Arrival_Time": self.arrival_time,
        }
