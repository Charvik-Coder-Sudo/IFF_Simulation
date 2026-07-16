"""Reusable statistics helpers, plus detection/authentication/track
breakdown rows and their CSV writers (Phase 10).

Purpose:
    Implements the small numeric helpers every other `analysis` module
    reuses (`mean`, `population_stdev`, `min_max`, `safe_divide`), plus
    three per-category breakdown tables: `DetectionStatistics` (one row
    per `IFFMode`), `AuthenticationStatistics` (one row per Mode 5
    level), and `TrackStatistics` (one row per track, completed or
    still active).

Inputs:
    A `PipelineRunRecord`.

Outputs:
    `list[DetectionStatistics]` / `list[AuthenticationStatistics]` /
    `list[TrackStatistics]`, and `write_detection_statistics_csv` /
    `write_authentication_statistics_csv` / `write_track_statistics_csv`
    writing `detection_statistics.csv` / `authentication_statistics.csv`
    / `track_statistics.csv`.

Engineering explanation:
    Every helper here is a pure function of its inputs -- no
    estimation, no randomness, no mutation of anything on
    `PipelineRunRecord`. `TrackStatistics` necessarily has fewer
    available fields for still-active tracks than for completed
    (lost) ones: `IFFTrack` (Phase 8) never accumulated lifetime
    counters the way `TrackSummary` (Phase 8.5) does for a *completed*
    track, so an active track's row reports its current range/signal
    strength (a single instantaneous value, not a true running average)
    and leaves `Duration`/`Replies_Received`/`Replies_Missed` as `None`
    -- this is a limitation of the data available from the existing
    pipeline, not something this analysis phase invents a number for.
"""

from __future__ import annotations

import csv
import statistics as _statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..sensors.iff import IFFMode, MeasurementStatus, TrackStatus
from .run_record import PipelineRunRecord


def mean(values: list) -> float:
    """Arithmetic mean, or 0.0 for an empty sequence (never raises)."""
    return sum(values) / len(values) if values else 0.0


def population_stdev(values: list) -> float:
    """Population standard deviation, or 0.0 for fewer than 2 samples."""
    return _statistics.pstdev(values) if len(values) >= 2 else 0.0


def min_max(values: list) -> tuple:
    """`(min, max)`, or `(0.0, 0.0)` for an empty sequence."""
    return (min(values), max(values)) if values else (0.0, 0.0)


def safe_divide(numerator: float, denominator: float) -> float:
    """`numerator / denominator`, or 0.0 if `denominator` is 0 (never raises)."""
    return numerator / denominator if denominator else 0.0


def _valid_measurements(record: PipelineRunRecord):
    """Every VALID real-origin measurement this run (excludes NO_REPLY/
    GARBLED, and excludes false-alarm/fruited measurements -- those are
    handled separately by callers that specifically want them)."""
    return [
        tick.real_measurement
        for tick in record.tick_results
        if tick.real_measurement is not None and tick.real_measurement.reply_status == MeasurementStatus.VALID
    ]


# ---------------------------------------------------------------------------
# Detection statistics (one row per IFFMode, plus an ALL aggregate row)
# ---------------------------------------------------------------------------


DETECTION_STATISTICS_CSV_COLUMNS = [
    "Mode", "Interrogations", "Expected_Replies", "Correct_Replies",
    "Detection_Probability", "Average_Range_M", "Maximum_Range_M", "Average_Signal_Strength",
]


@dataclass(frozen=True, slots=True)
class DetectionStatistics:
    """Detection outcome breakdown for one `IFFMode` (or "ALL")."""

    mode: str
    interrogations: int
    expected_replies: int
    correct_replies: int
    detection_probability: float
    average_range_m: float
    maximum_range_m: float
    average_signal_strength: float

    def to_csv_row(self) -> dict:
        return {
            "Mode": self.mode,
            "Interrogations": self.interrogations,
            "Expected_Replies": self.expected_replies,
            "Correct_Replies": self.correct_replies,
            "Detection_Probability": self.detection_probability,
            "Average_Range_M": self.average_range_m,
            "Maximum_Range_M": self.maximum_range_m,
            "Average_Signal_Strength": self.average_signal_strength,
        }


def _detection_statistics_for(mode_label: str, record: PipelineRunRecord, mode_filter) -> DetectionStatistics:
    interrogations = [i for i in record.interrogations if mode_filter(i.mode)]
    indices = [idx for idx, i in enumerate(record.interrogations) if mode_filter(i.mode)]
    expected = [idx for idx in indices if record.replies[idx] is not None]
    correct = [
        idx for idx in expected
        if record.tick_results[idx].real_measurement is not None
        and record.tick_results[idx].real_measurement.reply_status == MeasurementStatus.VALID
    ]
    ranges = [record.tick_results[idx].real_measurement.range_m for idx in correct]
    strengths = [
        record.tick_results[idx].real_measurement.signal_strength
        for idx in correct
        if record.tick_results[idx].real_measurement.signal_strength is not None
    ]
    _, max_range = min_max(ranges)
    return DetectionStatistics(
        mode=mode_label,
        interrogations=len(interrogations),
        expected_replies=len(expected),
        correct_replies=len(correct),
        detection_probability=safe_divide(len(correct), len(expected)),
        average_range_m=mean(ranges),
        maximum_range_m=max_range,
        average_signal_strength=mean(strengths),
    )


def compute_detection_statistics(record: PipelineRunRecord) -> list[DetectionStatistics]:
    """One `DetectionStatistics` row per `IFFMode` present, plus an "ALL" row."""
    rows = [
        _detection_statistics_for(mode.value, record, lambda m, mode=mode: m == mode)
        for mode in IFFMode
    ]
    rows.append(_detection_statistics_for("ALL", record, lambda m: True))
    return rows


def write_detection_statistics_csv(rows: Iterable[DetectionStatistics], output_path: Path | str) -> Path:
    return _write_csv((r.to_csv_row() for r in rows), DETECTION_STATISTICS_CSV_COLUMNS, output_path)


# ---------------------------------------------------------------------------
# Authentication statistics (one row per Mode 5 level, plus an ALL_MODE5 row)
# ---------------------------------------------------------------------------


AUTHENTICATION_STATISTICS_CSV_COLUMNS = [
    "Mode", "Mode5_Replies", "Authenticated", "Failed", "Authentication_Success_Rate",
]


@dataclass(frozen=True, slots=True)
class AuthenticationStatistics:
    """Authentication outcome breakdown for one Mode 5 level (or "ALL_MODE5")."""

    mode: str
    mode5_replies: int
    authenticated: int
    failed: int
    authentication_success_rate: float

    def to_csv_row(self) -> dict:
        return {
            "Mode": self.mode,
            "Mode5_Replies": self.mode5_replies,
            "Authenticated": self.authenticated,
            "Failed": self.failed,
            "Authentication_Success_Rate": self.authentication_success_rate,
        }


def _authentication_statistics_for(mode_label: str, measurements: list) -> AuthenticationStatistics:
    authenticated = sum(1 for m in measurements if m.authentication_result)
    failed = len(measurements) - authenticated
    return AuthenticationStatistics(
        mode=mode_label,
        mode5_replies=len(measurements),
        authenticated=authenticated,
        failed=failed,
        authentication_success_rate=safe_divide(authenticated, len(measurements)),
    )


def compute_authentication_statistics(record: PipelineRunRecord) -> list[AuthenticationStatistics]:
    """One `AuthenticationStatistics` row per Mode 5 level, plus "ALL_MODE5"."""
    valid = _valid_measurements(record)
    mode5_levels = [IFFMode.MODE5_L1, IFFMode.MODE5_L2]
    rows = [
        _authentication_statistics_for(level.value, [m for m in valid if m.mode == level])
        for level in mode5_levels
    ]
    rows.append(_authentication_statistics_for("ALL_MODE5", [m for m in valid if m.mode in mode5_levels]))
    return rows


def write_authentication_statistics_csv(rows: Iterable[AuthenticationStatistics], output_path: Path | str) -> Path:
    return _write_csv((r.to_csv_row() for r in rows), AUTHENTICATION_STATISTICS_CSV_COLUMNS, output_path)


# ---------------------------------------------------------------------------
# Track statistics (one row per track, completed or still active)
# ---------------------------------------------------------------------------


TRACK_STATISTICS_CSV_COLUMNS = [
    "Track_ID", "Aircraft_ID", "Track_Status", "Ever_Confirmed", "Duration",
    "Replies_Received", "Replies_Missed", "Average_Range_M", "Average_Signal_Strength",
]


@dataclass(frozen=True, slots=True)
class TrackStatistics:
    """One row describing a single track's outcome, completed or active."""

    track_id: int
    aircraft_id: str
    track_status: str
    ever_confirmed: bool
    duration: float | None
    replies_received: int | None
    replies_missed: int | None
    average_range_m: float | None
    average_signal_strength: float | None

    def to_csv_row(self) -> dict:
        return {
            "Track_ID": self.track_id,
            "Aircraft_ID": self.aircraft_id,
            "Track_Status": self.track_status,
            "Ever_Confirmed": self.ever_confirmed,
            "Duration": self.duration if self.duration is not None else "",
            "Replies_Received": self.replies_received if self.replies_received is not None else "",
            "Replies_Missed": self.replies_missed if self.replies_missed is not None else "",
            "Average_Range_M": self.average_range_m if self.average_range_m is not None else "",
            "Average_Signal_Strength": (
                self.average_signal_strength if self.average_signal_strength is not None else ""
            ),
        }


def compute_track_statistics(record: PipelineRunRecord) -> list[TrackStatistics]:
    """One `TrackStatistics` row per completed (lost) track, then per
    still-active track. See module docstring for why active-track rows
    have fewer populated fields."""
    rows = [
        TrackStatistics(
            track_id=summary.track_id,
            aircraft_id=summary.aircraft_id,
            track_status=summary.final_track_status.value,
            ever_confirmed=summary.confirmed_time > 0.0,
            duration=summary.duration,
            replies_received=summary.replies_received,
            replies_missed=summary.replies_missed,
            average_range_m=summary.avg_range_m,
            average_signal_strength=summary.avg_signal_strength,
        )
        for summary in record.completed_track_summaries
    ]
    rows.extend(
        TrackStatistics(
            track_id=track.track_id,
            aircraft_id=track.aircraft_id,
            track_status=track.track_status.value,
            ever_confirmed=track.track_status == TrackStatus.CONFIRMED,
            duration=None,
            replies_received=None,
            replies_missed=None,
            average_range_m=track.range_m,
            average_signal_strength=track.signal_strength,
        )
        for track in record.active_tracks
    )
    return rows


def write_track_statistics_csv(rows: Iterable[TrackStatistics], output_path: Path | str) -> Path:
    return _write_csv((r.to_csv_row() for r in rows), TRACK_STATISTICS_CSV_COLUMNS, output_path)


def _write_csv(rows: Iterable[dict], columns: list[str], output_path: Path | str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_path
