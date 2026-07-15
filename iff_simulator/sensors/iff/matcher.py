"""Matches a reply to its interrogation, or determines a timeout occurred.

Purpose:
    Implements `MatchResult` and `ReplyMatcher`, which take one
    `InterrogationMessage` and the `ReplyMessage` (if any) the
    transponder produced for it, propagate that reply, and produce
    exactly one outcome: matched (a `PropagatedReply` within its
    timeout window) or timed out (`propagated_reply=None`).

Inputs:
    An `InterrogationMessage`, an optional `ReplyMessage`, and the
    Ownship/target positions to propagate against.

Outputs:
    A `MatchResult`.

Engineering explanation:
    Matching uses Sequence Number, Target ID, Mode, and Ownship ID —
    exactly the four fields the spec names — as a defensive integrity
    check: every `ReplyMessage` generator (Phase 6) already copies
    these fields verbatim from its source interrogation, so a mismatch
    should never occur in practice, but a mismatched reply is still
    treated as "no reply for this interrogation" rather than trusted.
    The timeout decision reuses `Receiver.is_timed_out` (checked
    against the reply's own computed `arrival_time`) rather than
    re-deriving the mode-to-timeout mapping a second time.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...domain import Vector3
from .interrogation import InterrogationMessage
from .propagation import PropagatedReply, ReplyPropagation
from .receiver import Receiver
from .reply import ReplyMessage


@dataclass(frozen=True, slots=True)
class MatchResult:
    """The outcome of matching one interrogation to at most one reply.

    Purpose:
        Carry the single (interrogation, reply-or-none) pairing
        `ModeDecoder` needs to build exactly one `DecodedIFFMeasurement`
        per interrogation.

    Inputs:
        Constructed exclusively by `ReplyMatcher.match`.

    Outputs:
        Consumed by `ModeDecoder.decode`.
    """

    interrogation: InterrogationMessage
    """The interrogation this result answers."""

    propagated_reply: PropagatedReply | None
    """The matched, on-time `PropagatedReply`, or `None` if the
    transponder never replied, the reply didn't match this
    interrogation's identity, or it arrived too late (timed out)."""

    timed_out: bool
    """True whenever `propagated_reply` is None — kept as an explicit
    field (rather than relying on callers to check for None) so a
    "no reply" outcome is unambiguous even if a future field is added
    to this dataclass."""


class ReplyMatcher:
    """Matches a reply to its interrogation, applying identity checks and timeout.

    Purpose:
        The single place "does this interrogation have a valid,
        on-time reply" is decided.

    Inputs:
        propagation: a `ReplyPropagation` (dependency-injected;
            defaults to a real one).
        receiver: a `Receiver` (dependency-injected; defaults to a new
            one) — every propagated reply is still recorded into it,
            so `Receiver`'s ordering/buffering is genuinely exercised
            even though `match()` itself does not need to wait across
            simulation ticks to reach a decision.

    Outputs:
        `match(interrogation, reply, ownship_position, target_position) -> MatchResult`.

    Engineering explanation:
        Deterministic: given the same interrogation, reply (or None),
        and positions, always produces the same `MatchResult` — no
        waiting, no external clock dependency, no hidden state.
    """

    def __init__(
        self,
        propagation: ReplyPropagation | None = None,
        receiver: Receiver | None = None,
    ) -> None:
        self.propagation = propagation or ReplyPropagation()
        self.receiver = receiver or Receiver()

    def match(
        self,
        interrogation: InterrogationMessage,
        reply: ReplyMessage | None,
        ownship_position: Vector3,
        target_position: Vector3,
    ) -> MatchResult:
        """Match one interrogation to its reply (if any).

        Inputs:
            interrogation: the `InterrogationMessage` awaiting a reply.
            reply: the `ReplyMessage` the transponder produced for it,
                or `None` if the transponder did not reply at all
                (dead/incapable/mode-disabled — see `AirborneTransponder`).
            ownship_position, target_position: `Vector3` positions at
                reply time, for propagation.
        Outputs:
            A `MatchResult`: matched (propagated_reply set,
            timed_out=False) or not (propagated_reply=None,
            timed_out=True).
        """
        if reply is None or not self._identity_matches(reply, interrogation):
            return MatchResult(interrogation=interrogation, propagated_reply=None, timed_out=True)

        propagated = self.propagation.propagate(reply, ownship_position, target_position)
        self.receiver.receive(propagated)

        if self.receiver.is_timed_out(interrogation, propagated.arrival_time):
            return MatchResult(interrogation=interrogation, propagated_reply=None, timed_out=True)

        return MatchResult(interrogation=interrogation, propagated_reply=propagated, timed_out=False)

    @staticmethod
    def _identity_matches(reply: ReplyMessage, interrogation: InterrogationMessage) -> bool:
        """Sequence Number, Target ID, Mode, Ownship ID must all match."""
        return (
            reply.interrogation_sequence == interrogation.sequence_number
            and reply.target_id == interrogation.target_id
            and reply.mode == interrogation.mode
            and reply.ownship_id == interrogation.ownship_id
        )
