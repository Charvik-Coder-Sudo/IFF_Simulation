"""Timing jitter on processing delay, propagation delay, and arrival time
(Phase 9 Part 8).

Purpose:
    Implements `jitter_processing_delay`, which perturbs a `ReplyMessage`'s
    `processing_delay` before propagation, and `JitteredReplyPropagation`,
    a `ReplyPropagation` subclass that perturbs the resulting
    `propagation_delay_us`/`arrival_time` after propagation. Together
    they add uniform +/- jitter to all three timing quantities the spec
    names, without modifying `ReplyPropagation`/`ReplyMessage` themselves.

Inputs:
    A `ReplyMessage` (or the real `ReplyPropagation`'s output), a jitter
    bound in microseconds, and the shared seeded `random.Random`.

Outputs:
    A jittered `ReplyMessage` copy / a jittered `PropagatedReply`.

Engineering explanation:
    `JitteredReplyPropagation` subclasses (rather than modifies)
    `ReplyPropagation`: it calls `super().propagate()` to get the exact
    same deterministic distance/signal-strength computation every other
    caller relies on, then perturbs only the timing fields of the
    result via `dataclasses.replace`. This is additive extension, not a
    redesign of the completed `ReplyPropagation` class -- every existing
    caller of plain `ReplyPropagation` is completely unaffected.
"""

from __future__ import annotations

import dataclasses
import random

from ...domain import Vector3
from .propagation import PropagatedReply, ReplyPropagation
from .reply import ReplyMessage

_MICROSECONDS_PER_SECOND = 1_000_000.0


def jitter_processing_delay(reply: ReplyMessage, jitter_us: float, rng: random.Random) -> ReplyMessage:
    """Return a copy of `reply` with uniform +/- jitter added to its
    `processing_delay`.

    Inputs:
        reply: the `ReplyMessage` to perturb (unmodified; frozen).
        jitter_us: the +/- bound, microseconds. 0.0 adds no jitter
            (`rng.uniform(0.0, 0.0) == 0.0` always).
        rng: the shared seeded RNG.

    Outputs:
        A new `ReplyMessage` with only `processing_delay` changed.
    """
    if jitter_us == 0.0:
        return reply
    delta = rng.uniform(-jitter_us, jitter_us)
    return dataclasses.replace(reply, processing_delay=reply.processing_delay + delta)


class JitteredReplyPropagation(ReplyPropagation):
    """A `ReplyPropagation` that adds uniform +/- jitter to propagation
    delay and arrival time.

    Purpose:
        Extend (never modify) `ReplyPropagation` with Part 8's timing
        jitter, while keeping `propagate()`'s distance/signal-strength
        computation byte-identical to the base class.

    Inputs:
        jitter_us: the +/- bound applied to `propagation_delay_us` (and
            therefore `arrival_time`), microseconds. 0.0 (default)
            reproduces the base class's output exactly.
        rng: the shared seeded RNG.
        Remaining constructor args are forwarded to `ReplyPropagation`.

    Outputs:
        `propagate(...)` -> a jittered `PropagatedReply`.
    """

    def __init__(self, rng: random.Random, jitter_us: float = 0.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.rng = rng
        self.jitter_us = jitter_us

    def propagate(
        self,
        reply: ReplyMessage,
        ownship_position: Vector3,
        target_position: Vector3,
    ) -> PropagatedReply:
        propagated = super().propagate(reply, ownship_position, target_position)
        if self.jitter_us == 0.0:
            return propagated
        delta_us = self.rng.uniform(-self.jitter_us, self.jitter_us)
        return dataclasses.replace(
            propagated,
            propagation_delay_us=propagated.propagation_delay_us + delta_us,
            arrival_time=propagated.arrival_time + delta_us / _MICROSECONDS_PER_SECOND,
        )
