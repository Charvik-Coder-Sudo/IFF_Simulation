"""Writes IFFMeasurementReports to a deterministically-ordered CSV file.

Purpose:
    Implements `ReportWriter`, which serializes a collection of
    `IFFMeasurementReport` to CSV, sorted by (Time, Track_ID), using
    exactly the column names this phase specifies.

Inputs:
    An iterable of `IFFMeasurementReport`.

Outputs:
    A CSV file (conventionally named `iff_track_file.csv`).

Engineering explanation:
    Sorting by `(time, track_id)` — rather than trusting insertion
    order — guarantees the file's row order is a pure function of its
    contents, not of whatever order the caller happened to generate
    reports in.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from ...domain import Vector3
from .report_generator import IFFMeasurementReport

CSV_COLUMNS = [
    "Time",
    "Track_ID",
    "Aircraft_ID",
    "Ownship_ID",
    "Range",
    "Azimuth",
    "Elevation",
    "Relative_Velocity",
    "Mode",
    "Reply_Status",
    "ModeS_Address",
    "Authentication_Result",
    "Friend_Foe_Status",
    "Track_Quality",
    "Track_Status",
    "Sequence_Number",
    "Signal_Strength",
    "Propagation_Delay",
]


def _format_relative_velocity(velocity: Vector3 | None) -> str:
    """Serialize a Vector3 as "x;y;z", or "" if None.

    Engineering reasoning:
        A semicolon (not comma) separator keeps the value readable as a
        single CSV field without relying on the CSV writer's
        comma-quoting behavior.
    """
    if velocity is None:
        return ""
    return f"{velocity.x};{velocity.y};{velocity.z}"


def _report_to_row(report: IFFMeasurementReport) -> dict:
    return {
        "Time": report.time,
        "Track_ID": report.track_id,
        "Aircraft_ID": report.aircraft_id,
        "Ownship_ID": report.ownship_id,
        "Range": report.range_m,
        "Azimuth": report.azimuth_deg,
        "Elevation": report.elevation_deg,
        "Relative_Velocity": _format_relative_velocity(report.relative_velocity),
        "Mode": report.mode.value,
        "Reply_Status": report.reply_status.value,
        "ModeS_Address": report.mode_s_address or "",
        "Authentication_Result": report.authentication_result,
        "Friend_Foe_Status": report.friend_foe_status.value,
        "Track_Quality": report.track_quality,
        "Track_Status": report.track_status.value,
        "Sequence_Number": report.sequence_number,
        "Signal_Strength": report.signal_strength,
        "Propagation_Delay": report.propagation_delay,
    }


class ReportWriter:
    """Writes IFFMeasurementReports to a deterministically-ordered CSV file.

    Purpose:
        The single place report rows are ordered and serialized.

    Inputs:
        `write(reports, output_path)`.

    Outputs:
        The resolved output path.

    Engineering explanation:
        Sorts by `(time, track_id)` before writing — see module
        docstring — so the file's contents are fully determined by the
        input reports, independent of call order.
    """

    def write(self, reports: Iterable[IFFMeasurementReport], output_path: Path | str) -> Path:
        """Write reports to CSV, sorted by (Time, Track_ID).

        Inputs:
            reports: any iterable of `IFFMeasurementReport`.
            output_path: destination CSV path. Parent directories are
                created automatically.
        Outputs:
            The resolved output path.
        """
        ordered_reports = sorted(reports, key=lambda report: (report.time, report.track_id))

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for report in ordered_reports:
                writer.writerow(_report_to_row(report))
        return output_path
