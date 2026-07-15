"""FIFO queue of transmitted interrogation messages, plus CSV export.

Purpose:
    Implements `InterrogationQueue`, the single-threaded FIFO buffer of
    `InterrogationMessage`s the scheduler transmits, awaiting whatever
    future stage consumes them (a receiver/reply model — not
    implemented yet). Also provides `write_interrogations_csv`, which
    serializes a collection of messages to `interrogations.csv`.

Inputs:
    `InterrogationMessage` instances, enqueued one at a time by
    `InterrogationScheduler`.

Outputs:
    Messages dequeued/peeked in strict first-in-first-out order; a CSV
    file when `write_interrogations_csv` is called.

Engineering explanation:
    Backed by `collections.deque`, which gives O(1) `enqueue`
    (`append`) and `dequeue` (`popleft`) — no list-shifting cost, and
    no threading/locking, per this phase's "no threading" constraint
    (this queue is not meant to be shared across threads).
"""

from __future__ import annotations

import csv
from collections import deque
from pathlib import Path
from typing import Iterable

from .interrogation import InterrogationMessage

_CSV_COLUMNS = [
    "Time",
    "Sequence",
    "Ownship_ID",
    "Target_ID",
    "Mode",
    "UF",
    "Range",
    "Azimuth",
    "Elevation",
    "Closing_Velocity",  # Phase 8.5 Part 1: appended, does not disturb prior columns.
    "Relative_Velocity",  # Phase 8.5 Part 1: appended, does not disturb prior columns.
]


class InterrogationQueue:
    """A single-threaded FIFO queue of InterrogationMessage.

    Purpose:
        Hold transmitted interrogations in transmission order until a
        future stage consumes them.

    Inputs:
        `enqueue(message)`.

    Outputs:
        `dequeue()`, `peek()`, `clear()`, `__len__`/`length()`.

    Engineering explanation:
        Strictly FIFO: `dequeue()` always returns the oldest
        not-yet-dequeued message, matching transmission order (which is
        itself ordered by strictly increasing sequence number).
    """

    def __init__(self) -> None:
        self._messages: deque[InterrogationMessage] = deque()

    def enqueue(self, message: InterrogationMessage) -> None:
        """Add a message to the back of the queue."""
        self._messages.append(message)

    def dequeue(self) -> InterrogationMessage:
        """Remove and return the oldest message.

        Raises:
            IndexError: if the queue is empty.
        """
        if not self._messages:
            raise IndexError("Cannot dequeue from an empty InterrogationQueue.")
        return self._messages.popleft()

    def peek(self) -> InterrogationMessage | None:
        """Return the oldest message without removing it, or None if empty."""
        return self._messages[0] if self._messages else None

    def clear(self) -> None:
        """Remove every message from the queue."""
        self._messages.clear()

    def length(self) -> int:
        """Return the number of messages currently queued."""
        return len(self)

    def __len__(self) -> int:
        return len(self._messages)


def write_interrogations_csv(
    messages: Iterable[InterrogationMessage], output_path: Path | str
) -> Path:
    """Write a collection of InterrogationMessage to interrogations.csv.

    Purpose:
        Persist transmitted interrogations for later inspection, in the
        exact column order the spec requires.
    Inputs:
        messages: any iterable of `InterrogationMessage` (e.g. a list
            collected while running the scheduler, or the current
            contents of an `InterrogationQueue`).
        output_path: destination CSV path. Parent directories are
            created automatically.
    Outputs:
        The resolved output path.
    Engineering reasoning:
        Uses the standard-library `csv` module rather than pandas:
        this is a flat list-of-dataclasses export with no aggregation,
        so a lighter-weight writer keeps this package's dependencies
        minimal, matching the Geometry package's precedent of avoiding
        pandas anywhere it is not genuinely needed.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for message in messages:
            writer.writerow(message.to_csv_row())
    return output_path
