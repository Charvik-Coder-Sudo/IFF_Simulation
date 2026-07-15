"""The airborne transponder: decides whether to reply, and to whom.

Purpose:
    Implements `AirborneTransponder`, which receives one
    `InterrogationMessage`, decides — using only Ground Truth — whether
    the target aircraft would reply at all, and if so, delegates to
    `ReplyBuilder` to produce the `ReplyMessage`. This is the final
    stage this phase implements: no RF, no propagation, no pulse/
    bitstream simulation, no cryptography.

Inputs:
    A `Scenario` (Ground Truth — the only place this class reads
    aircraft data from) and a `ReplyBuilder` (dependency-injected).

Outputs:
    `receive(interrogation) -> ReplyMessage | None`.

Engineering explanation:
    `AirborneTransponder` never estimates or propagates aircraft
    position: every fact it uses (`alive`, `iff_capability`,
    `mode_data`, kinematics) comes directly from
    `Scenario.get_aircraft`/`get_state`, i.e. Ground Truth as already
    recorded. "Target selected" (one of the spec's four reply-decision
    inputs) is not re-checked here: an `InterrogationMessage` only ever
    exists for a target `TargetSelector` (Phase 4) already selected, so
    that condition is satisfied by construction, not by a redundant
    check.
"""

from __future__ import annotations

from ...domain import Aircraft, Scenario
from .interrogation import InterrogationMessage
from .mode import IFFMode
from .reply import ReplyMessage
from .reply_builder import ReplyBuilder


class AirborneTransponder:
    """Decides whether an aircraft replies to an interrogation, and builds the reply.

    Purpose:
        The single entry point for "given this interrogation, does the
        target's transponder reply, and with what."

    Inputs:
        scenario: the `Scenario` (Ground Truth) to read the target
            aircraft's identity and live state from.
        reply_builder: a `ReplyBuilder` (dependency-injected; defaults
            to a real one).
        enable_logging: Phase 8.5 Part 7 — when True, every generated
            `ReplyMessage` also accumulates in `self.log`, for later
            export via `csv_logging.write_replies_csv`. Default False;
            `receive()`'s return value is identical either way.

    Outputs:
        `receive(interrogation) -> ReplyMessage | None`.

    Engineering explanation:
        Runs in O(1): one `Scenario.get_aircraft`/`get_state` lookup
        (both O(1) dict lookups) plus a fixed number of boolean checks
        — no search, no iteration over other aircraft.
    """

    def __init__(
        self,
        scenario: Scenario,
        reply_builder: ReplyBuilder | None = None,
        enable_logging: bool = False,
    ) -> None:
        self.scenario = scenario
        self.reply_builder = reply_builder or ReplyBuilder()
        self.enable_logging = enable_logging
        self.log: list[ReplyMessage] = []

    def receive(self, interrogation: InterrogationMessage) -> ReplyMessage | None:
        """Receive one interrogation and return a reply, or None.

        Reply Decision (in order):
            1. Aircraft Alive == False -> No Reply.
            2. IFF_Capable == False -> No Reply.
            3. The requested mode is not enabled on this aircraft -> No Reply.
            4. Otherwise -> a ReplyMessage is generated (an
               authentication failure, checked inside `ReplyBuilder`'s
               Mode 5 path, still produces a reply — with
               `authenticated=False` and `reply_status=FAILED_AUTH` —
               it is not the same thing as "no reply").

        Inputs:
            interrogation: the `InterrogationMessage` to respond to.
        Outputs:
            A `ReplyMessage`, or `None` if the transponder does not reply.
        """
        aircraft = self.scenario.get_aircraft(interrogation.target_id)
        aircraft_state = self.scenario.get_state(interrogation.target_id)

        if not aircraft_state.alive:
            return None
        if not self._is_iff_capable(aircraft):
            return None
        if not self._is_mode_enabled(aircraft, interrogation.mode):
            return None

        reply = self.reply_builder.build(interrogation, aircraft, aircraft_state)
        if self.enable_logging:
            self.log.append(reply)
        return reply

    @staticmethod
    def _is_iff_capable(aircraft: Aircraft) -> bool:
        """True unless `Aircraft.iff_capability` is still its "UNKNOWN"
        default (the same convention `World.iff_capable_targets()` and
        `DefaultSelectionPolicy` already use — no new field invented)."""
        return aircraft.iff_capability not in (None, "UNKNOWN")

    @staticmethod
    def _is_mode_enabled(aircraft: Aircraft, mode: IFFMode) -> bool:
        """True if `mode` appears in `Aircraft.mode_data["enabled_modes"]`.

        Engineering explanation:
            `mode_data` is Phase 1.5's placeholder dict for exactly
            this kind of mode-specific configuration; it defaults to an
            empty list here, meaning every mode is disabled for any
            aircraft that has never had its `mode_data` populated —
            expected and correct for all current recorded Ground Truth.
        """
        enabled_modes = aircraft.mode_data.get("enabled_modes", [])
        return mode.value in enabled_modes
