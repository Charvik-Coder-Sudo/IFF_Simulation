"""CSV export for the replies/decoded/tracks logs (Phase 8.5 Part 7).

Purpose:
    Implements `write_replies_csv`, `write_decoded_csv`, and
    `write_tracks_csv` — the remaining three of the six `logs/*.csv`
    files this phase asks every pipeline stage to optionally support.
    (`interrogations.csv` already existed via
    `interrogation_queue.write_interrogations_csv`; `reports.csv` via
    `ReportWriter.write`; `track_summary.csv` via
    `track_summary.write_track_summary_csv`.)

Inputs:
    Collections of `ReplyMessage` / `DecodedIFFMeasurement` / `IFFTrack`
    — typically each stage's own `.log` list, populated only when that
    stage was constructed with `enable_logging=True`.

Outputs:
    A CSV file per collection.

Engineering explanation:
    "Logging enabled through enable_logging=True. When disabled, no
    files generated" is satisfied one level up, by each stage
    (`AirborneTransponder`, `ModeDecoder`, `IFFTrackManager`) only ever
    populating its `.log` list when `enable_logging=True` — if a caller
    never enables logging, `.log` stays empty and calling these
    functions on it (or not calling them at all) produces no meaningful
    file. These functions themselves are unconditional, deterministic
    CSV writers, mirroring `write_interrogations_csv`'s existing shape.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from .measurement import DecodedIFFMeasurement
from .receiver_statistics import RECEIVER_STATISTICS_CSV_COLUMNS, ReceiverStatistics
from .reply import ReplyMessage
from .track import IFFTrack

REPLIES_CSV_COLUMNS = [
    "Time", "Sequence", "Ownship_ID", "Target_ID", "Mode", "Reply_Type",
    "Reply_Status", "Authenticated", "Authentication_Status", "ModeS_Address",
    "Mode5_Level", "Processing_Delay",
]

DECODED_CSV_COLUMNS = [
    "Time", "Sequence", "Target_ID", "Ownship_ID", "Mode", "Range", "Azimuth", "Elevation",
    "Closing_Velocity", "Relative_Velocity", "ICAO_Address", "Authentication_Result",
    "Authentication_Status", "Identity", "Mission", "Reply_Status", "Processing_Delay",
    "Propagation_Delay", "Signal_Strength", "Arrival_Time",
]

TRACKS_CSV_COLUMNS = [
    "Time", "Track_ID", "Aircraft_ID", "Ownship_ID", "Range", "Azimuth", "Elevation",
    "Closing_Velocity", "Relative_Velocity", "Mode", "Reply_Status", "ModeS_Address",
    "Authentication_Result", "Authentication_Status", "Friend_Foe_Status", "Track_Status",
    "Track_Quality", "Confidence", "Last_Update_Time", "Sequence_Number", "Reply_Type",
    "Signal_Strength", "Propagation_Delay",
]


def _write_csv(rows: Iterable[dict], columns: list[str], output_path: Path | str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_path


def write_replies_csv(replies: Iterable[ReplyMessage], output_path: Path | str) -> Path:
    """Write a collection of ReplyMessage to replies.csv."""
    return _write_csv((r.to_csv_row() for r in replies), REPLIES_CSV_COLUMNS, output_path)


def write_decoded_csv(measurements: Iterable[DecodedIFFMeasurement], output_path: Path | str) -> Path:
    """Write a collection of DecodedIFFMeasurement to decoded.csv."""
    return _write_csv((m.to_csv_row() for m in measurements), DECODED_CSV_COLUMNS, output_path)


def write_tracks_csv(tracks: Iterable[IFFTrack], output_path: Path | str) -> Path:
    """Write a collection of IFFTrack to tracks.csv."""
    return _write_csv((t.to_csv_row() for t in tracks), TRACKS_CSV_COLUMNS, output_path)


def write_receiver_statistics_csv(statistics: ReceiverStatistics, output_path: Path | str) -> Path:
    """Write one ReceiverStatistics snapshot to receiver_statistics.csv
    (Phase 9 Part 9), one data row following the header."""
    return _write_csv([statistics.to_csv_row()], RECEIVER_STATISTICS_CSV_COLUMNS, output_path)
