"""Logical Mode 5 authentication check — no cryptography.

Purpose:
    Implements `AuthenticationEngine`, which decides whether a Mode 5
    reply should be marked `authenticated=True`, using only declarative
    Ground Truth flags. No AES, no STANAG 4193/4479 implementation, no
    key exchange — a purely logical stand-in for "would this aircraft's
    crypto fill have authenticated." Also implements
    `classify_friendly_status`, which turns an authentication result
    (plus the aircraft's Ground Truth `identity`) into this project's
    BLUE/RED/NEUTRAL/UNKNOWN vocabulary.

Inputs:
    The target's `Aircraft` (its `mode_data` dict carries the three
    logical flags `authenticate()` reads; `identity` is what
    `classify_friendly_status` maps).

Outputs:
    `bool` — whether the reply authenticates. `str` — one of
    BLUE/RED/NEUTRAL/UNKNOWN.

Engineering explanation:
    `Aircraft.mode_data` is the exact placeholder dict Phase 1.5
    introduced for "raw Mode 1/2/3/4/5/S data" and left empty pending
    real IFF logic — this is that logic, reading three keys from it
    rather than inventing a new storage mechanism. Since every real
    Ground Truth aircraft's `mode_data` is still `{}` by default (no
    phase has ever populated it), `authenticate()` deterministically
    returns `False` for all current recorded data — expected and
    correct, matching this project's established pattern (e.g. Phase
    4's `iff_capable_targets()`).
"""

from __future__ import annotations

from enum import Enum, unique

from ...domain import Aircraft
from .mode import IFFMode

#: The mode_data["authentication_status"] value that means "authenticated."
_AUTHENTICATED_STATUS = "AUTHENTICATED"


@unique
class AuthenticationResult(Enum):
    """Semantic authentication outcome for one reply.

    Purpose:
        Phase 8.5's Part 2 replacement for representing authentication
        as a bare boolean: a three-state outcome that also captures
        "this mode has no authentication concept at all" (Mode S),
        distinct from "it was attempted and failed" (Mode 5).

    Engineering explanation:
        Added *alongside* the existing boolean fields
        (`ReplyMessage.authenticated`, `DecodedIFFMeasurement`/`IFFTrack`/
        `IFFMeasurementReport`'s `authentication_result: bool`) rather
        than replacing them in place: this phase's own success criteria
        require "all existing tests continue to pass" and "previous
        outputs remain backward compatible unless explicitly extended,"
        and those fields are already named/typed as booleans in
        completed, tested phases. `authentication_status` is the new,
        semantically richer field new consumers should prefer; the
        boolean fields remain for exact backward compatibility.
    """

    AUTHENTICATED = "AUTHENTICATED"
    FAILED = "FAILED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


def derive_authentication_status(mode: IFFMode, authenticated: bool) -> AuthenticationResult:
    """Derive the semantic AuthenticationResult for a reply.

    Purpose:
        Implement exactly the three rules Part 2 specifies, as a pure,
        reusable, independently-testable function.

    Inputs:
        mode: the `IFFMode` this reply answers.
        authenticated: the existing boolean authentication outcome
            (always False for Mode S).

    Outputs:
        `AuthenticationResult.NOT_APPLICABLE` for Mode S (it has no
        authentication mechanism in this simulator); otherwise
        `AUTHENTICATED` or `FAILED` for Mode 5, mirroring `authenticated`.

    Engineering reasoning:
        A pure function of already-known fields — no new state, no
        estimation, no cryptography.
    """
    if mode == IFFMode.MODE_S:
        return AuthenticationResult.NOT_APPLICABLE
    return AuthenticationResult.AUTHENTICATED if authenticated else AuthenticationResult.FAILED

#: Backward-compatible mapping from this project's earlier, informal
#: FRIEND/FOE-style identity strings (still used by older test fixtures
#: and any hand-set `Aircraft.identity`) to the BLUE/RED/NEUTRAL/UNKNOWN
#: vocabulary. `Aircraft.identity` itself is never rewritten — this is
#: a read-only lookup used only when deriving a reply's friendly-status
#: field.
_LEGACY_IDENTITY_TO_STATUS = {
    "FRIEND": "BLUE",
    "FRIENDLY": "BLUE",
    "FOE": "RED",
    "HOSTILE": "RED",
    "ENEMY": "RED",
    "NEUTRAL": "NEUTRAL",
    "CIVIL": "NEUTRAL",
    "UNKNOWN": "UNKNOWN",
}


def classify_friendly_status(identity: str, authenticated: bool) -> str:
    """Map an identity string + authentication result to BLUE/RED/NEUTRAL/UNKNOWN.

    Purpose:
        Provide the single, shared place `ModeSReplyGenerator` and
        `Mode5ReplyGenerator` derive a reply's friendly-status field
        from, so both use the same vocabulary and the same rule.

    Inputs:
        identity: `Aircraft.identity` (Ground Truth's own field —
            never modified by this function).
        authenticated: whether Mode 5 authentication succeeded for
            this reply (always `False` for Mode S, which has no
            authentication step).

    Outputs:
        One of `"BLUE"`, `"RED"`, `"NEUTRAL"`, `"UNKNOWN"`.

    Engineering reasoning:
        Successful Mode 5 authentication is treated as a sufficient
        (not exclusive) condition for BLUE: a cryptographically
        authenticated aircraft is always classified BLUE regardless of
        its declared identity string, since that is the strongest
        friendliness signal available. Otherwise, a legacy identity
        string already declaring FRIEND/FOE/etc. is mapped to its
        BLUE/RED/NEUTRAL equivalent (backward compatibility with
        earlier phases/tests); anything unrecognized defaults to
        `"UNKNOWN"` rather than guessing.
    """
    if authenticated:
        return "BLUE"
    return _LEGACY_IDENTITY_TO_STATUS.get(identity.upper(), "UNKNOWN")


class AuthenticationEngine:
    """Decides whether a target's Mode 5 reply authenticates.

    Purpose:
        Implement exactly the three-condition rule this phase
        specifies: authenticated only if the aircraft's declared
        authentication status is "AUTHENTICATED", AND its Mode 5
        capability is enabled, AND it has a crypto key present.

    Inputs:
        aircraft: the target's `Aircraft`, whose `mode_data` dict holds
            `"authentication_status"`, `"mode5_enabled"`, and
            `"crypto_key_present"` (all absent -> not authenticated).

    Outputs:
        `bool`.

    Engineering explanation:
        A pure function of `Aircraft.mode_data` — no cryptography, no
        randomness, no state held across calls. The same `Aircraft`
        always authenticates the same way.
    """

    def authenticate(self, aircraft: Aircraft) -> bool:
        """Return whether this aircraft's Mode 5 reply should authenticate.

        Mathematics:
            authenticated =
                (mode_data["authentication_status"] == "AUTHENTICATED")
                AND mode_data["mode5_enabled"]
                AND mode_data["crypto_key_present"]
            (all three keys default to a falsy value if absent.)
        """
        status = aircraft.mode_data.get("authentication_status", "UNKNOWN")
        mode5_enabled = bool(aircraft.mode_data.get("mode5_enabled", False))
        crypto_key_present = bool(aircraft.mode_data.get("crypto_key_present", False))
        return status == _AUTHENTICATED_STATUS and mode5_enabled and crypto_key_present
