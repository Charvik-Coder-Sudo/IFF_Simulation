"""Mode S reply generation.

Purpose:
    Implements `ModeSPayload` (the structured Mode S reply content) and
    `ModeSReplyGenerator`, which builds a `ReplyMessage` for a Mode S
    `InterrogationMessage` directly from the target's Ground Truth —
    never estimating or inventing any value. Also implements
    `compute_icao_address`, the deterministic 24-bit hex ICAO address
    assignment this phase's "Small Improvements" replaced the earlier
    `"ICAO-{aircraft_id}"` placeholder with.

Inputs:
    An `InterrogationMessage` (mode == IFFMode.MODE_S) and the target's
    `Aircraft` + `AircraftState`.

Outputs:
    A `ReplyMessage` with a `ModeSPayload`.

Engineering explanation:
    Mode S has no authentication step in this simulator (real Mode S
    is not a cryptographically secure protocol), so every Mode S reply
    is unconditionally `authenticated=False`, `reply_status=OK` — there
    is no failure path here, unlike Mode 5.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from ...domain import Aircraft, AircraftState
from .authentication import AuthenticationResult, classify_friendly_status
from .interrogation import InterrogationMessage
from .reply import ReplyMessage, ReplyPayload, ReplyStatus, ReplyType
from .uplink_format import UplinkFormat

#: Real Mode S protocol pairing, at the logical-label level only:
#: an all-call/Comm-B *interrogation* format elicits the matching
#: *reply* format (UF11->DF11, UF20->DF20, UF21->DF21).
_REPLY_TYPE_BY_UPLINK_FORMAT: dict[UplinkFormat, ReplyType] = {
    UplinkFormat.UF11: ReplyType.DF11,
    UplinkFormat.UF20: ReplyType.DF20,
    UplinkFormat.UF21: ReplyType.DF21,
}

PROCESSING_DELAY_US = 50.0
"""Deterministic Mode S transponder processing delay, microseconds."""

_ICAO_BLOCK_BASE = 0xA00000
"""Top hex nibble fixed at 'A': a deliberate, clearly-simulator-owned
24-bit address block, distinct from any real-world allocation."""

_ICAO_BLOCK_MASK = 0xFFFFF
"""20 bits available below the fixed top nibble."""

_TRAILING_DIGITS = re.compile(r"(\d+)$")


def compute_icao_address(aircraft_id: str) -> str:
    """Deterministic 24-bit hexadecimal ICAO address for an aircraft_id.

    Purpose:
        Replace the earlier `f"ICAO-{aircraft_id}"` placeholder with a
        realistic 24-bit hex ICAO-style address, while guaranteeing the
        same aircraft always gets the same address and no address is
        ever produced randomly.

    Inputs:
        aircraft_id: the aircraft's Scenario aircraft_id (e.g.
            "TARGET_1", or any other string used in tests).

    Outputs:
        A 6-character uppercase hex string, e.g. `"A00001"`.

    Mathematics:
        If `aircraft_id` ends in digits (true for every real Ground
        Truth aircraft, e.g. "TARGET_1"), that number is used directly:
        `address = 0xA00000 + (number & 0xFFFFF)`, giving exactly
        A00001, A00002, A00003, ... for TARGET_1, TARGET_2, TARGET_3.
        Otherwise (e.g. "OWNSHIP", or an arbitrary test id with no
        trailing digits), a stable 32-bit value is derived from a
        SHA-256 digest of the id string and masked the same way.

    Engineering reasoning:
        Never uses Python's built-in `hash()`: that is randomized
        per-process (`PYTHONHASHSEED`) and would violate "the same
        aircraft must always receive the same ICAO address" across
        separate runs. `hashlib.sha256` is stable across processes and
        Python versions, which a per-process-random hash is not.
    """
    match = _TRAILING_DIGITS.search(aircraft_id)
    if match:
        number = int(match.group(1))
    else:
        digest = hashlib.sha256(aircraft_id.encode("utf-8")).hexdigest()
        number = int(digest[:8], 16)
    icao_value = _ICAO_BLOCK_BASE + (number & _ICAO_BLOCK_MASK)
    return f"{icao_value:06X}"


@dataclass(frozen=True, slots=True)
class ModeSPayload(ReplyPayload):
    """Structured Mode S reply content.

    Purpose:
        Carry the logical equivalent of a Mode S reply's addressable
        content: ICAO address, altitude, identity, capability, and the
        logical downlink format number — as data, never as an encoded
        bitstream.

    Inputs:
        Built exclusively by `ModeSReplyGenerator` from Ground Truth.

    Outputs:
        Attached to a `ReplyMessage` as its `payload`.

    Engineering explanation:
        `altitude_m` is read directly from the target's own
        `AircraftState.position.z` — the aircraft's *own* recorded
        altitude — never derived from `InterrogationMessage`'s
        Ownship-relative range/azimuth/elevation, which describe a
        completely different (relative) quantity. This is exactly what
        "never estimate aircraft position; only read Ground Truth"
        requires.
    """

    icao_address: str
    """Logical Mode S address (see `mode_s_address` on `ReplyMessage`)."""

    altitude_m: float
    """The target's own altitude, meters — `AircraftState.position.z`,
    read directly from Ground Truth."""

    identity: str
    """The target's friendly-status classification: one of
    BLUE/RED/NEUTRAL/UNKNOWN, derived from `Aircraft.identity` via
    `classify_friendly_status` (Mode S never authenticates, so this is
    always based on the legacy identity mapping, never forced to BLUE
    the way an authenticated Mode 5 reply's would be)."""

    capability: str
    """The target's declared IFF capability, copied verbatim from
    `Aircraft.iff_capability`."""

    df_number: str
    """Logical downlink format number (matches `ReplyMessage.reply_type.value`)."""


class ModeSReplyGenerator:
    """Builds a ReplyMessage for a Mode S interrogation.

    Purpose:
        The single place Mode S reply content is assembled, from
        Ground Truth alone.

    Inputs:
        `generate(interrogation, aircraft, aircraft_state)`.

    Outputs:
        A `ReplyMessage`.

    Engineering explanation:
        Deterministic and stateless: the same
        (interrogation, aircraft, aircraft_state) triple always
        produces the same `ReplyMessage`, satisfying this phase's
        "reproducible from Ground Truth + Interrogation alone"
        requirement.
    """

    def generate(
        self,
        interrogation: InterrogationMessage,
        aircraft: Aircraft,
        aircraft_state: AircraftState,
    ) -> ReplyMessage:
        """Build a Mode S ReplyMessage.

        Inputs:
            interrogation: the Mode S `InterrogationMessage` being answered.
            aircraft: the target's static `Aircraft` metadata.
            aircraft_state: the target's live `AircraftState`.
        Outputs:
            A `ReplyMessage` with `mode_s_address` set and a `ModeSPayload`.
        """
        mode_s_address = compute_icao_address(aircraft.aircraft_id)
        reply_type = _REPLY_TYPE_BY_UPLINK_FORMAT[interrogation.uplink_format]

        payload = ModeSPayload(
            icao_address=mode_s_address,
            altitude_m=aircraft_state.position.z,
            identity=classify_friendly_status(aircraft.identity, authenticated=False),
            capability=aircraft.iff_capability,
            df_number=reply_type.value,
        )

        return ReplyMessage(
            reply_id=interrogation.sequence_number,
            time=interrogation.time,
            interrogation_sequence=interrogation.sequence_number,
            ownship_id=interrogation.ownship_id,
            target_id=interrogation.target_id,
            mode=interrogation.mode,
            reply_type=reply_type,
            reply_status=ReplyStatus.OK,
            authenticated=False,
            mode_s_address=mode_s_address,
            mode1=None,
            mode2=None,
            mode3A=None,
            modeC=None,
            mode5_level=None,
            payload=payload,
            processing_delay=PROCESSING_DELAY_US,
            authentication_status=AuthenticationResult.NOT_APPLICABLE,
        )
