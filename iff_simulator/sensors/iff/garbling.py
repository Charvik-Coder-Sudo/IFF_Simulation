"""Garbling detection (Phase 9 Part 3).

Purpose:
    Implements `detect_garbled`, the pure function
    `ReceiverEffectsPipeline` uses to decide which replies in an
    arrival-ordered batch are unreadable because another reply (of any
    origin -- real, false-alarm, or fruited) arrived too close in time.

Inputs:
    A list of `PropagatedReply`, already sorted by `arrival_time` (as
    `Receiver.pop_ready` guarantees), and the configured garble window,
    seconds.

Outputs:
    A `set` of `id(propagated_reply)` for every reply involved in a
    garbling collision -- both replies in a colliding pair, since a real
    garbled reply is unreadable for either party, not just the later one.

Engineering explanation:
    A pure, stateless function of its inputs alone -- no RNG, no
    estimation. Only adjacent pairs (in arrival order) are compared:
    since the batch is already sorted by arrival_time, if replies i and
    i+2 were close enough to garble, i+1 (in between them in time) would
    already have triggered a collision with one of its neighbors, so
    comparing only adjacent pairs is sufficient and keeps this O(n).
"""

from __future__ import annotations


def detect_garbled(sorted_batch: list, garble_window_s: float) -> set:
    """Return the set of `id(propagated_reply)` for every reply in
    `sorted_batch` involved in a garbling collision.

    Inputs:
        sorted_batch: propagated replies sorted by `arrival_time`
            ascending (any origin).
        garble_window_s: two replies whose arrival times differ by less
            than this are both garbled. `<= 0.0` never garbles (the
            default/off configuration).

    Outputs:
        A `set` of `id(...)` values; empty if nothing garbled.
    """
    garbled_ids: set = set()
    if garble_window_s <= 0.0:
        return garbled_ids
    for earlier, later in zip(sorted_batch, sorted_batch[1:]):
        if abs(later.arrival_time - earlier.arrival_time) < garble_window_s:
            garbled_ids.add(id(earlier))
            garbled_ids.add(id(later))
    return garbled_ids
