"""A buffer of received replies, ordered by arrival time then sequence number.

Purpose:
    Implements `ReceiverBuffer`, which holds `PropagatedReply` objects
    and always yields them (via `peek`/`pop`) in the order they
    actually arrive — earliest `arrival_time` first, ties broken by the
    reply's own sequence number.

Inputs:
    `PropagatedReply` instances, inserted one at a time.

Outputs:
    Replies retrieved in strict arrival order.

Engineering explanation:
    Backed by a binary heap (`heapq`), giving O(log n) `insert`/`pop`
    and O(1) `peek` — no packet loss, no capacity limit ("no overflow"):
    every inserted reply is retained until popped or `clear()`ed. A
    monotonic insertion counter breaks any remaining ties (identical
    arrival_time *and* identical sequence number, which should not
    happen given Phase 5's unique sequence numbers, but is handled
    defensively) so the heap never needs to compare two
    `PropagatedReply` objects directly.
"""

from __future__ import annotations

import heapq
import itertools

from .propagation import PropagatedReply


class ReceiverBuffer:
    """Orders received replies by (arrival_time, sequence_number).

    Purpose:
        Guarantee replies are always retrieved in true arrival order,
        regardless of the order they were inserted in.

    Inputs:
        `insert(propagated_reply)`.

    Outputs:
        `pop()`, `peek()`, `clear()`, `__len__`.

    Engineering explanation:
        "No packet loss, no overflow" — insertion always succeeds and
        nothing is ever dropped except by an explicit `pop()`/`clear()`.
    """

    def __init__(self) -> None:
        self._heap: list[tuple[float, int, int, PropagatedReply]] = []
        self._counter = itertools.count()

    def insert(self, propagated_reply: PropagatedReply) -> None:
        """Insert a propagated reply, ordered by (arrival_time, sequence_number)."""
        sequence_number = propagated_reply.reply.reply_id
        heapq.heappush(
            self._heap,
            (propagated_reply.arrival_time, sequence_number, next(self._counter), propagated_reply),
        )

    def pop(self) -> PropagatedReply:
        """Remove and return the earliest-arriving buffered reply.

        Raises:
            IndexError: if the buffer is empty.
        """
        if not self._heap:
            raise IndexError("Cannot pop from an empty ReceiverBuffer.")
        *_, propagated_reply = heapq.heappop(self._heap)
        return propagated_reply

    def peek(self) -> PropagatedReply | None:
        """Return the earliest-arriving buffered reply without removing it,
        or None if the buffer is empty."""
        return self._heap[0][-1] if self._heap else None

    def clear(self) -> None:
        """Remove every buffered reply."""
        self._heap.clear()

    def __len__(self) -> int:
        return len(self._heap)
