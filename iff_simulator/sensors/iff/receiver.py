"""The interrogator's receiver: acceptance, timeout, ordering, forwarding.

Purpose:
    Implements `Receiver`, which accepts propagated replies into a
    `ReceiverBuffer`, decides whether a given interrogation's reply
    window has timed out, and forwards replies that have actually
    arrived (by simulated time) in arrival order. No decoding happens
    here — `Receiver` never looks inside a reply's payload.

Inputs:
    `PropagatedReply` instances (via `receive`), and the current
    simulation time (for `pop_ready`/`is_timed_out`).

Outputs:
    Replies that have arrived, in arrival order (`pop_ready`); a
    timeout determination (`is_timed_out`).

Engineering explanation:
    Timeout durations are deterministic constants per mode (Mode S:
    500 microseconds; Mode 5: 700 microseconds), never estimated or
    randomized.
"""

from __future__ import annotations

from .interrogation import InterrogationMessage
from .mode import IFFMode
from .propagation import PropagatedReply
from .receiver_buffer import ReceiverBuffer

_MICROSECONDS_PER_SECOND = 1_000_000.0

TIMEOUT_SECONDS_BY_MODE: dict[IFFMode, float] = {
    IFFMode.MODE_S: 500.0 / _MICROSECONDS_PER_SECOND,
    IFFMode.MODE5_L1: 700.0 / _MICROSECONDS_PER_SECOND,
    IFFMode.MODE5_L2: 700.0 / _MICROSECONDS_PER_SECOND,
}
"""Deterministic reply-timeout window per mode, converted to the same
time unit as InterrogationMessage.time (seconds)."""


class Receiver:
    """Accepts, times out, orders, and forwards propagated replies.

    Purpose:
        Own the four responsibilities the spec assigns to the
        receiver: reply acceptance, reply timeout, reply ordering, and
        reply forwarding — nothing about decoding.

    Inputs:
        buffer: a `ReceiverBuffer` (dependency-injected; defaults to a
            new, empty one).

    Outputs:
        `receive(propagated_reply)`, `is_timed_out(interrogation, as_of_time)`,
        `pop_ready(current_time)`.

    Engineering explanation:
        `is_timed_out` takes an explicit `as_of_time` rather than
        reading a clock itself — `Receiver` holds no `SimulationClock`
        of its own (reusing the one `World` already owns, per this
        phase's "do not create another clock" constraint); callers
        pass whichever time they need the determination made against
        (typically a reply's own `arrival_time`, or the live
        `World.current_time()`).
    """

    def __init__(self, buffer: ReceiverBuffer | None = None) -> None:
        self.buffer = buffer if buffer is not None else ReceiverBuffer()

    def receive(self, propagated_reply: PropagatedReply) -> None:
        """Accept a propagated reply (reply acceptance + ordering, via ReceiverBuffer)."""
        self.buffer.insert(propagated_reply)

    @staticmethod
    def is_timed_out(interrogation: InterrogationMessage, as_of_time: float) -> bool:
        """Return whether `interrogation`'s reply-timeout window has elapsed
        as of `as_of_time`.

        Mathematics:
            deadline = interrogation.time + TIMEOUT_SECONDS_BY_MODE[interrogation.mode]
            timed_out = as_of_time > deadline
        """
        deadline = interrogation.time + TIMEOUT_SECONDS_BY_MODE[interrogation.mode]
        return as_of_time > deadline

    def pop_ready(self, current_time: float) -> list[PropagatedReply]:
        """Pop and return every buffered reply that has already arrived
        (arrival_time <= current_time), in arrival order.

        Purpose:
            Reply ordering + reply forwarding: only replies that have
            actually arrived by `current_time` are ever forwarded.
        """
        ready: list[PropagatedReply] = []
        while self.buffer.peek() is not None and self.buffer.peek().arrival_time <= current_time:
            ready.append(self.buffer.pop())
        return ready

    def __len__(self) -> int:
        return len(self.buffer)
