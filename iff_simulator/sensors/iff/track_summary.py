"""Full-lifetime summary statistics for a completed (lost) IFF track.

Purpose:
    Defines `TrackSummary` (one row of `track_summary.csv`) and
    `write_track_summary_csv`, which serializes a collection of them.
    `IFFTrackManager` builds a `TrackSummary` the instant a track
    transitions to Lost, from statistics it has been accumulating over
    that track's entire lifetime (not from the bounded 20-item history
    deque, which is deliberately too short to answer "what was this
    track's total duration/average range").

Inputs:
    Built exclusively by `IFFTrackManager` when a track is lost.

Outputs:
    `track_summary.csv`, for debugging, visualization, and statistics —
    never used to drive any decision inside the simulator itself.

Engineering explanation:
    One row per *completed* (lost) track only — a track still active
    when the simulation ends has no `Track_End_Time` yet and is not
    included here (it remains inspectable live via
    `IFFTrackManager.get_active_tracks()`).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .track import TrackStatus

TRACK_SUMMARY_CSV_COLUMNS = [
    "Track_ID",
    "Aircraft_ID",
    "Ownship_ID",
    "Track_Start_Time",
    "Track_End_Time",
    "Duration",
    "Replies_Received",
    "Replies_Missed",
    "Tentative_Time",
    "Confirmed_Time",
    "Lost_Time",
    "Maximum_Range",
    "Minimum_Range",
    "Average_Range",
    "Maximum_Signal_Strength",
    "Average_Signal_Strength",
    "Final_Track_Status",
]


@dataclass(frozen=True, slots=True)
class TrackSummary:
    """One completed track's full-lifetime summary statistics.

    Purpose:
        Carry exactly the fields this phase's Part 6 schema lists.

    Inputs:
        Constructed exclusively by `IFFTrackManager` when a track is lost.

    Outputs:
        Consumed by `write_track_summary_csv`.

    Engineering explanation:
        `tentative_time`/`confirmed_time` partition `duration` by which
        lifecycle status the track was in; `lost_time` is always 0.0 —
        "Lost" is a terminal, instantaneous transition in this
        simulator (a track is deleted the same tick it is marked Lost),
        so there is no elapsed time to have been *spent* in that state.
        It is still a field here (rather than omitted) for schema
        completeness and because a future phase might introduce a
        non-instantaneous Lost state (e.g. a coast period).
    """

    track_id: int
    aircraft_id: str
    ownship_id: str
    track_start_time: float
    track_end_time: float
    duration: float
    replies_received: int
    replies_missed: int
    tentative_time: float
    confirmed_time: float
    lost_time: float
    max_range_m: float
    min_range_m: float
    avg_range_m: float
    max_signal_strength: float
    avg_signal_strength: float
    final_track_status: TrackStatus

    def to_csv_row(self) -> dict:
        """Return this summary as a dict keyed by the track_summary.csv column names."""
        return {
            "Track_ID": self.track_id,
            "Aircraft_ID": self.aircraft_id,
            "Ownship_ID": self.ownship_id,
            "Track_Start_Time": self.track_start_time,
            "Track_End_Time": self.track_end_time,
            "Duration": self.duration,
            "Replies_Received": self.replies_received,
            "Replies_Missed": self.replies_missed,
            "Tentative_Time": self.tentative_time,
            "Confirmed_Time": self.confirmed_time,
            "Lost_Time": self.lost_time,
            "Maximum_Range": self.max_range_m,
            "Minimum_Range": self.min_range_m,
            "Average_Range": self.avg_range_m,
            "Maximum_Signal_Strength": self.max_signal_strength,
            "Average_Signal_Strength": self.avg_signal_strength,
            "Final_Track_Status": self.final_track_status.value,
        }


def write_track_summary_csv(summaries: Iterable[TrackSummary], output_path: Path | str) -> Path:
    """Write a collection of TrackSummary to track_summary.csv.

    Inputs:
        summaries: any iterable of `TrackSummary`.
        output_path: destination CSV path. Parent directories are
            created automatically.
    Outputs:
        The resolved output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACK_SUMMARY_CSV_COLUMNS)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary.to_csv_row())
    return output_path
