"""Receiver-level statistics accumulation and reporting (Phase 9 Part 9).

Purpose:
    Implements `ReceiverStatistics` (an immutable snapshot) and
    `ReceiverStatisticsCollector` (the mutable accumulator
    `ReceiverEffectsPipeline` feeds every tick), plus
    `RECEIVER_STATISTICS_CSV_COLUMNS` for `csv_logging.write_receiver_statistics_csv`.

Inputs:
    Per-tick outcomes reported by `ReceiverEffectsPipeline`: whether a
    real reply was received/lost/garbled, whether a false alarm or
    fruited reply was generated, the Pd value rolled, the signal
    strength and total delay of any propagated reply.

Outputs:
    A `ReceiverStatistics` snapshot (`collector.snapshot()`), and bounded
    per-tick time series (`load_history`, `garbled_history`,
    `false_reply_history`, `fruited_history`) for Part 10's timeline
    plots.

Engineering explanation:
    Mirrors `track_manager._TrackState`'s accumulator pattern: a single
    mutable bookkeeping object that is never itself exposed, only ever
    read via an immutable snapshot. Purely additive counting -- no
    estimation, filtering, or smoothing of any kind.
"""

from __future__ import annotations

from dataclasses import dataclass, field

RECEIVER_STATISTICS_CSV_COLUMNS = [
    "Replies_Received", "Replies_Lost", "Replies_Garbled", "Replies_Fruited",
    "False_Replies", "Average_Detection_Probability", "Average_Signal_Strength",
    "Average_Delay_Us", "Receiver_Load",
]


@dataclass(frozen=True, slots=True)
class ReceiverStatistics:
    """Immutable snapshot of receiver-level statistics for one run.

    Purpose:
        Carry exactly the fields Part 9 asks for: Replies Received,
        Replies Lost, Replies Garbled, Replies Fruited, False Replies,
        Detection Probability, Average Signal Strength, Average Delay,
        Receiver Load.
    """

    replies_received: int
    """Real replies successfully decoded as VALID."""

    replies_lost: int
    """Real reply attempts that did not result in a VALID measurement,
    for any reason other than garbling (failed Pd roll, below
    sensitivity threshold, timed out, or dropped by saturation)."""

    replies_garbled: int
    """Replies (of any origin) marked GARBLED this run."""

    replies_fruited: int
    """Fruited replies decoded this run."""

    false_replies: int
    """False-alarm replies generated this run."""

    average_detection_probability: float
    """Running average of the Pd value rolled against for every real
    reply attempt (0.0 if no attempts were made)."""

    average_signal_strength: float
    """Running average signal strength across every propagated reply
    (real, false, or fruited) this run (0.0 if none)."""

    average_delay_us: float
    """Running average total delay (processing + propagation),
    microseconds, across every propagated reply this run (0.0 if none)."""

    receiver_load: float
    """Average number of replies (all origins) processed per tick."""

    def to_csv_row(self) -> dict:
        """Return this snapshot as a dict, for `receiver_statistics.csv`."""
        return {
            "Replies_Received": self.replies_received,
            "Replies_Lost": self.replies_lost,
            "Replies_Garbled": self.replies_garbled,
            "Replies_Fruited": self.replies_fruited,
            "False_Replies": self.false_replies,
            "Average_Detection_Probability": self.average_detection_probability,
            "Average_Signal_Strength": self.average_signal_strength,
            "Average_Delay_Us": self.average_delay_us,
            "Receiver_Load": self.receiver_load,
        }


@dataclass(slots=True)
class ReceiverStatisticsCollector:
    """Mutable accumulator for `ReceiverStatistics` -- never exposed
    directly; always read via `snapshot()`."""

    replies_received: int = 0
    replies_lost: int = 0
    replies_garbled: int = 0
    replies_fruited: int = 0
    false_replies: int = 0

    _pd_sum: float = 0.0
    _pd_count: int = 0
    _signal_sum: float = 0.0
    _signal_count: int = 0
    _delay_sum: float = 0.0
    _delay_count: int = 0
    _tick_count: int = 0
    _replies_this_run: int = 0

    load_history: list = field(default_factory=list)
    """List of `(time, replies_processed_this_tick)`."""

    garbled_history: list = field(default_factory=list)
    """List of times at which a reply was marked GARBLED."""

    false_reply_history: list = field(default_factory=list)
    """List of times at which a false alarm was generated."""

    fruited_history: list = field(default_factory=list)
    """List of times at which a fruited reply was decoded."""

    def record_pd_roll(self, pd: float) -> None:
        """Record one Pd value rolled against for a real reply attempt."""
        self._pd_sum += pd
        self._pd_count += 1

    def record_signal_strength(self, signal_strength: float) -> None:
        """Record one propagated reply's signal strength (any origin)."""
        self._signal_sum += signal_strength
        self._signal_count += 1

    def record_delay(self, delay_us: float) -> None:
        """Record one propagated reply's total delay, microseconds (any origin)."""
        self._delay_sum += delay_us
        self._delay_count += 1

    def record_received(self) -> None:
        self.replies_received += 1

    def record_lost(self) -> None:
        self.replies_lost += 1

    def record_garbled(self, time: float) -> None:
        self.replies_garbled += 1
        self.garbled_history.append(time)

    def record_fruited(self, time: float) -> None:
        self.replies_fruited += 1
        self.fruited_history.append(time)

    def record_false_reply(self, time: float) -> None:
        self.false_replies += 1
        self.false_reply_history.append(time)

    def record_tick_load(self, time: float, replies_processed: int) -> None:
        """Record how many replies (all origins) were processed this tick,
        for the "Receiver Load vs Time" plot and the running load average."""
        self._tick_count += 1
        self._replies_this_run += replies_processed
        self.load_history.append((time, replies_processed))

    def snapshot(self) -> ReceiverStatistics:
        """Build the immutable `ReceiverStatistics` snapshot for this run so far."""
        avg_pd = self._pd_sum / self._pd_count if self._pd_count else 0.0
        avg_signal = self._signal_sum / self._signal_count if self._signal_count else 0.0
        avg_delay = self._delay_sum / self._delay_count if self._delay_count else 0.0
        load = self._replies_this_run / self._tick_count if self._tick_count else 0.0
        return ReceiverStatistics(
            replies_received=self.replies_received,
            replies_lost=self.replies_lost,
            replies_garbled=self.replies_garbled,
            replies_fruited=self.replies_fruited,
            false_replies=self.false_replies,
            average_detection_probability=avg_pd,
            average_signal_strength=avg_signal,
            average_delay_us=avg_delay,
            receiver_load=load,
        )
