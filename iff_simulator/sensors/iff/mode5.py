"""Mode 5 Level 1 / Level 2 reply generation.

Purpose:
    Implements `Mode5Level1Payload`, `Mode5Level2Payload` (the
    structured content for each supported level), and
    `Mode5ReplyGenerator`, which builds a `ReplyMessage` for a Mode 5
    `InterrogationMessage` directly from the target's Ground Truth plus
    a logical authentication check. Also implements
    `compute_mission_code`, the deterministic per-aircraft mission-type
    assignment this phase's "Small Improvements" replaced the earlier
    bare `"UNKNOWN"` mission default with.

Inputs:
    An `InterrogationMessage` (mode == MODE5_L1 or MODE5_L2), the
    target's `Aircraft` + `AircraftState`, and an `AuthenticationEngine`.

Outputs:
    A `ReplyMessage` with a `Mode5Level1Payload` or `Mode5Level2Payload`.

Scope:
    Level 1 and Level 2 only. Level 3/4/5 are out of scope for this
    simulator and are not represented here at all.

Engineering explanation:
    Unlike Mode S, a Mode 5 reply is still generated even when
    authentication fails (`reply_status=FAILED_AUTH`,
    `authenticated=False`) — per this phase's explicit "Reply Decision"
    rule, an authentication failure is not the same thing as "no
    reply."
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ...domain import Aircraft, AircraftState
from .authentication import AuthenticationEngine, classify_friendly_status, derive_authentication_status
from .interrogation import InterrogationMessage
from .mode import IFFMode
from .reply import ReplyMessage, ReplyPayload, ReplyStatus, ReplyType

PROCESSING_DELAY_US = 75.0
"""Deterministic Mode 5 transponder processing delay, microseconds."""

MISSION_TYPES = [
    "CAP",
    "AIR_DEFENCE",
    "STRIKE",
    "CAS",
    "SEAD",
    "ESCORT",
    "AEW",
    "TANKER",
    "RECON",
    "PATROL",
]
"""The realistic mission-type vocabulary `compute_mission_code` assigns
from, replacing the earlier generic "UNKNOWN"/placeholder mission default."""


def compute_mission_code(aircraft_id: str) -> str:
    """Deterministic mission-type assignment for an aircraft_id.

    Purpose:
        Give every aircraft a realistic, stable mission type (one of
        `MISSION_TYPES`) instead of a generic placeholder, without any
        randomness.

    Inputs:
        aircraft_id: the aircraft's Scenario aircraft_id.

    Outputs:
        One of `MISSION_TYPES`.

    Engineering reasoning:
        Uses a stable `hashlib.sha256` digest (never Python's
        randomized built-in `hash()`) modulo the vocabulary size, so
        the same aircraft always gets the same mission type, in every
        run and every process.
    """
    digest = hashlib.sha256(aircraft_id.encode("utf-8")).hexdigest()
    index = int(digest[:8], 16) % len(MISSION_TYPES)
    return MISSION_TYPES[index]


@dataclass(frozen=True, slots=True)
class Mode5Level1Payload(ReplyPayload):
    """Structured Mode 5 Level 1 reply content.

    Purpose:
        Carry the logical equivalent of a Mode 5 L1 reply: its own
        authentication result, mission code, platform ID, and friendly
        status.

    Engineering explanation:
        `mission_code`/`platform_id` are read from `Aircraft.mode_data`
        (Phase 1.5's placeholder dict for exactly this kind of
        mode-specific data), each falling back to a documented default
        when absent — never invented or randomized.
    """

    authentication_result: bool
    """Whether this reply authenticated (mirrors `ReplyMessage.authenticated`;
    carried in the payload too since it is itself part of a real L1 reply's content)."""

    mission_code: str
    """`Aircraft.mode_data["mission_code"]`, or this aircraft's
    deterministic `compute_mission_code` assignment if not set."""

    platform_id: str
    """`Aircraft.mode_data["platform_id"]`, or the aircraft's own
    aircraft_id if not set."""

    friendly_status: str
    """One of BLUE/RED/NEUTRAL/UNKNOWN, from `classify_friendly_status`
    — BLUE whenever this reply authenticated, otherwise derived from
    `Aircraft.identity`."""


@dataclass(frozen=True, slots=True)
class Mode5Level2Payload(ReplyPayload):
    """Structured Mode 5 Level 2 reply content.

    Purpose:
        Carry the logical equivalent of a Mode 5 L2 reply: platform
        address, mission, time, and additional status.

    Engineering explanation:
        `platform_address`/`mission`/`additional_status` are read from
        `Aircraft.mode_data`, each with a documented fallback default.
        `time` is copied verbatim from the interrogation/reply time —
        never recomputed. Payload content is plain data, not an
        encrypted bitstream, per this phase's explicit constraint.
    """

    platform_address: str
    """`Aircraft.mode_data["platform_address"]`, or the aircraft's own
    aircraft_id if not set."""

    mission: str
    """`Aircraft.mode_data["mission"]`, or this aircraft's deterministic
    `compute_mission_code` assignment if not set."""

    time: float
    """The interrogation/reply time this payload corresponds to."""

    additional_status: str
    """`Aircraft.mode_data["additional_status"]`, or "NONE" if not set."""


class Mode5ReplyGenerator:
    """Builds a ReplyMessage for a Mode 5 (Level 1 or Level 2) interrogation.

    Purpose:
        The single place Mode 5 reply content and authentication are
        assembled, from Ground Truth alone.

    Inputs:
        authentication_engine: the `AuthenticationEngine` to consult
            (dependency-injected; defaults to a real one).
        `generate(interrogation, aircraft, aircraft_state)`.

    Outputs:
        A `ReplyMessage`.

    Engineering explanation:
        Deterministic and stateless (aside from the injected, itself
        stateless `AuthenticationEngine`): the same
        (interrogation, aircraft, aircraft_state) triple always
        produces the same `ReplyMessage`.
    """

    def __init__(self, authentication_engine: AuthenticationEngine | None = None) -> None:
        self.authentication_engine = authentication_engine or AuthenticationEngine()

    def generate(
        self,
        interrogation: InterrogationMessage,
        aircraft: Aircraft,
        aircraft_state: AircraftState,
    ) -> ReplyMessage:
        """Build a Mode 5 ReplyMessage.

        Inputs:
            interrogation: the MODE5_L1 or MODE5_L2 `InterrogationMessage`
                being answered.
            aircraft: the target's static `Aircraft` metadata.
            aircraft_state: the target's live `AircraftState` (unused
                directly here — Mode 5 payloads carry no kinematic
                data — but accepted for a uniform generator interface
                with `ModeSReplyGenerator`).
        Outputs:
            A `ReplyMessage` with `mode5_level` set and a
            `Mode5Level1Payload` or `Mode5Level2Payload`.
        """
        authenticated = self.authentication_engine.authenticate(aircraft)
        reply_status = ReplyStatus.OK if authenticated else ReplyStatus.FAILED_AUTH

        if interrogation.mode == IFFMode.MODE5_L1:
            level = 1
            reply_type = ReplyType.MODE5_L1_REPLY
            payload: ReplyPayload = Mode5Level1Payload(
                authentication_result=authenticated,
                mission_code=aircraft.mode_data.get(
                    "mission_code", compute_mission_code(aircraft.aircraft_id)
                ),
                platform_id=aircraft.mode_data.get("platform_id", aircraft.aircraft_id),
                friendly_status=classify_friendly_status(aircraft.identity, authenticated),
            )
        else:
            level = 2
            reply_type = ReplyType.MODE5_L2_REPLY
            payload = Mode5Level2Payload(
                platform_address=aircraft.mode_data.get("platform_address", aircraft.aircraft_id),
                mission=aircraft.mode_data.get(
                    "mission", compute_mission_code(aircraft.aircraft_id)
                ),
                time=interrogation.time,
                additional_status=aircraft.mode_data.get("additional_status", "NONE"),
            )

        return ReplyMessage(
            reply_id=interrogation.sequence_number,
            time=interrogation.time,
            interrogation_sequence=interrogation.sequence_number,
            ownship_id=interrogation.ownship_id,
            target_id=interrogation.target_id,
            mode=interrogation.mode,
            reply_type=reply_type,
            reply_status=reply_status,
            authenticated=authenticated,
            mode_s_address=None,
            mode1=None,
            mode2=None,
            mode3A=None,
            modeC=None,
            mode5_level=level,
            payload=payload,
            processing_delay=PROCESSING_DELAY_US,
            authentication_status=derive_authentication_status(interrogation.mode, authenticated),
        )
