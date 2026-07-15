"""Immutable record of one transponder reply to an interrogation.

Purpose:
    Defines `ReplyMessage` — the logical reply the airborne transponder
    produces for one `InterrogationMessage` — plus the small supporting
    types (`ReplyType`, `ReplyStatus`, `ReplyPayload`) it is built from.

Inputs:
    Built exclusively by `ModeSReplyGenerator` / `Mode5ReplyGenerator`
    from an `InterrogationMessage` and the target's own Ground Truth
    (`Aircraft` + `AircraftState`); never estimates or invents data.

Outputs:
    Consumed by whatever later stage processes replies (a receiver/
    decoder — not implemented yet) and by test/inspection code.

Engineering explanation:
    Frozen (immutable), for the same reason every other per-instant
    record in this codebase is (`AircraftState` aside, which is
    intentionally mutable for a different reason — see its own
    docstring): a reply is a fact about one interrogation and must
    never be mutated after creation. `ReplyPayload` is an empty marker
    base class, not a real interface, defined here (rather than in
    `mode_s.py`/`mode5.py`) purely so `ReplyMessage.payload` can be
    meaningfully typed without `reply.py` importing from `mode_s.py`/
    `mode5.py` — which would create a circular import, since both of
    those modules import `ReplyMessage` from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique

from .authentication import AuthenticationResult
from .mode import IFFMode


class ReplyPayload:
    """Empty marker base class for the structured, mode-specific reply payload.

    Purpose:
        Give `ReplyMessage.payload` a real, shared type
        (`ModeSPayload`, `Mode5Level1Payload`, `Mode5Level2Payload` all
        subclass this) without `reply.py` depending on `mode_s.py`/
        `mode5.py`.

    Engineering explanation:
        Deliberately has no fields or methods of its own — payload
        content is entirely mode-specific and lives on the concrete
        subclasses in `mode_s.py`/`mode5.py`.
    """


@unique
class ReplyType(Enum):
    """Logical downlink-format-style label for a reply — no bit-field encoding.

    Purpose:
        Mirror Phase 5's `UplinkFormat` on the downlink side: a logical
        label for "what kind of reply this is," not a real Mode S
        Downlink Format bit pattern.

    Engineering explanation:
        Mode S values pair 1:1 with the `UplinkFormat` values Phase 5
        already supports (UF11->DF11, UF20->DF20, UF21->DF21), matching
        real Mode S protocol pairing at the logical level. Mode 5 has
        no equivalent "DF number" scheme in reality, so
        `MODE5_L1_REPLY`/`MODE5_L2_REPLY` are this simulator's own
        logical labels for those two reply kinds.
    """

    DF11 = "DF11"
    DF20 = "DF20"
    DF21 = "DF21"
    MODE5_L1_REPLY = "MODE5_L1_REPLY"
    MODE5_L2_REPLY = "MODE5_L2_REPLY"


@unique
class ReplyStatus(Enum):
    """Outcome status of a generated reply.

    Purpose:
        Distinguish a normally-generated reply from one whose Mode 5
        authentication failed (which is still generated, per the
        "Reply Decision" rules — a failed authentication is not the
        same thing as "no reply").

    Engineering explanation:
        Mode S replies are always `OK`: Mode S has no authentication
        mechanism in this simulator, so it has no failure path to this
        status (see `ModeSReplyGenerator`).

        Phase 8.5 Part 3 extends this enum with a realistic taxonomy of
        reply outcomes for later RF-level phases. `OK` and `FAILED_AUTH`
        (this simulator's original two values) are unchanged and remain
        the only values `ModeSReplyGenerator`/`Mode5ReplyGenerator` ever
        produce today. `VALID`/`NO_REPLY` are included here too only for
        vocabulary completeness with `MeasurementStatus` (which is the
        enum actually used for that VALID/NO_REPLY distinction at the
        measurement/track layer — see `measurement.py`); this class
        never assigns them. `TIMEOUT`, `GARBLED`, `FRUITED`,
        `LATE_REPLY`, `CRC_ERROR`, and `UNKNOWN_MODE` are structural
        placeholders only — no logic in this phase produces or
        interprets them; they exist purely so a future RF/receiver-realism
        phase can start assigning them without changing this enum's shape.
    """

    OK = "OK"
    FAILED_AUTH = "FAILED_AUTH"

    # --- Phase 8.5 Part 3: placeholders for future phases only ---
    VALID = "VALID"
    NO_REPLY = "NO_REPLY"
    TIMEOUT = "TIMEOUT"
    GARBLED = "GARBLED"
    FRUITED = "FRUITED"
    LATE_REPLY = "LATE_REPLY"
    CRC_ERROR = "CRC_ERROR"
    UNKNOWN_MODE = "UNKNOWN_MODE"


@dataclass(frozen=True, slots=True)
class ReplyMessage:
    """One logical transponder reply to an interrogation.

    Purpose:
        Carry everything about one reply: which interrogation it
        answers, its mode/type/status, whether it authenticated, and
        its structured (never string-encoded) payload.

    Inputs:
        Constructed by `ModeSReplyGenerator.generate` /
        `Mode5ReplyGenerator.generate`; not intended to be hand-built
        by callers.

    Outputs:
        Consumed by test/inspection code and any future receiver stage.

    Engineering explanation:
        `reply_id` and `interrogation_sequence` are both set to the
        source `InterrogationMessage.sequence_number` — deliberately
        *not* an independent counter. Phase 5 already guarantees
        interrogation sequence numbers are strictly monotonic and
        never reused, and at most one reply exists per interrogation,
        so reusing that same number keeps every reply's identity
        derivable purely from its interrogation (the "no hidden state"
        requirement) rather than depending on how many replies a given
        `AirborneTransponder` instance has produced before it.
        `mode1`/`mode2`/`mode3A`/`modeC` are structural placeholders
        only (always `None`): Phase 5 explicitly excluded those legacy
        modes from `IFFMode`, and no logic in this phase populates them
        either — they exist only so a later legacy-mode phase can add
        that logic without changing this dataclass's shape.
    """

    reply_id: int
    """Identifier for this reply; equal to the source interrogation's
    sequence number (see engineering explanation above)."""

    time: float
    """Simulation time this reply corresponds to (copied verbatim from
    the source InterrogationMessage — never recomputed)."""

    interrogation_sequence: int
    """The source InterrogationMessage's sequence_number, preserved verbatim."""

    ownship_id: str
    """Aircraft ID of the Ownship that sent the interrogation this replies to."""

    target_id: str
    """Aircraft ID of the aircraft replying (matches the interrogation's target_id)."""

    mode: IFFMode
    """The IFF mode this reply answers (copied verbatim from the interrogation)."""

    reply_type: ReplyType
    """Logical downlink-format-style label for this reply."""

    reply_status: ReplyStatus
    """OK, or FAILED_AUTH if Mode 5 authentication did not pass."""

    authenticated: bool
    """Whether this reply passed Mode 5 authentication. Always False for
    Mode S replies (no authentication mechanism exists for Mode S here).
    Kept for exact backward compatibility; prefer `authentication_status`
    (Phase 8.5) for new code — see its docstring."""

    mode_s_address: str | None
    """Logical Mode S address, derived from the target's aircraft_id.
    None for Mode 5 replies."""

    mode1: str | None
    """Legacy Mode 1 placeholder — always None; Mode 1 is out of scope."""

    mode2: str | None
    """Legacy Mode 2 placeholder — always None; Mode 2 is out of scope."""

    mode3A: str | None
    """Legacy Mode 3/A placeholder — always None; Mode 3/A is out of scope."""

    modeC: str | None
    """Legacy Mode C placeholder — always None; Mode C is out of scope."""

    mode5_level: int | None
    """1 or 2 for Mode 5 replies (matching the requested level); None
    for Mode S replies."""

    payload: ReplyPayload
    """Structured, mode-specific reply content: a `ModeSPayload`,
    `Mode5Level1Payload`, or `Mode5Level2Payload`."""

    processing_delay: float
    """Deterministic transponder processing delay, microseconds."""

    authentication_status: AuthenticationResult = AuthenticationResult.NOT_APPLICABLE
    """Phase 8.5 Part 2: the semantic authentication outcome (see
    `AuthenticationResult`). Defaults to `NOT_APPLICABLE` (Mode S's
    value) so every pre-existing `ReplyMessage(...)` construction that
    predates this field keeps working unmodified — this is purely
    additive, never a replacement of `authenticated`."""

    def to_csv_row(self) -> dict:
        """Return this reply as a dict, for Phase 8.5 Part 7's replies.csv logging.

        Outputs:
            dict with keys: Time, Sequence, Ownship_ID, Target_ID, Mode,
            Reply_Type, Reply_Status, Authenticated, Authentication_Status,
            ModeS_Address, Mode5_Level, Processing_Delay. `payload` is
            intentionally not flattened here (it is mode-specific
            structured data, not a flat CSV row).
        """
        return {
            "Time": self.time,
            "Sequence": self.interrogation_sequence,
            "Ownship_ID": self.ownship_id,
            "Target_ID": self.target_id,
            "Mode": self.mode.value,
            "Reply_Type": self.reply_type.value,
            "Reply_Status": self.reply_status.value,
            "Authenticated": self.authenticated,
            "Authentication_Status": self.authentication_status.value,
            "ModeS_Address": self.mode_s_address or "",
            "Mode5_Level": self.mode5_level if self.mode5_level is not None else "",
            "Processing_Delay": self.processing_delay,
        }
