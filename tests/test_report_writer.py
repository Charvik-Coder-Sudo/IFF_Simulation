"""Tests for ReportWriter."""

from __future__ import annotations

from pathlib import Path

from iff_simulator.domain import Vector3
from iff_simulator.sensors.iff import (
    CSV_COLUMNS,
    FriendFoeStatus,
    IFFMeasurementReport,
    IFFMode,
    MeasurementStatus,
    ReportWriter,
    TrackStatus,
)

ZERO = Vector3(0.0, 0.0, 0.0)


def _report(**overrides) -> IFFMeasurementReport:
    fields = dict(
        time=1.0,
        track_id=1,
        aircraft_id="T1",
        ownship_id="OWNSHIP",
        range_m=100.0,
        azimuth_deg=0.0,
        elevation_deg=0.0,
        relative_velocity=None,
        mode=IFFMode.MODE_S,
        reply_status=MeasurementStatus.VALID,
        mode_s_address="A00001",
        authentication_result=False,
        friend_foe_status=FriendFoeStatus.UNKNOWN,
        track_quality=0.3,
        track_status=TrackStatus.TENTATIVE,
        sequence_number=1,
        signal_strength=1.0,
        propagation_delay=1.0,
    )
    fields.update(overrides)
    return IFFMeasurementReport(**fields)


def test_csv_header_matches_exact_spec_columns(tmp_path: Path):
    output_path = tmp_path / "iff_track_file.csv"
    ReportWriter().write([_report()], output_path)
    header = output_path.read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == CSV_COLUMNS
    assert CSV_COLUMNS == [
        "Time", "Track_ID", "Aircraft_ID", "Ownship_ID", "Range", "Azimuth", "Elevation",
        "Relative_Velocity", "Mode", "Reply_Status", "ModeS_Address", "Authentication_Result",
        "Friend_Foe_Status", "Track_Quality", "Track_Status", "Sequence_Number",
        "Signal_Strength", "Propagation_Delay",
    ]


def test_rows_sorted_by_time_then_track_id(tmp_path: Path):
    reports = [
        _report(time=2.0, track_id=1, aircraft_id="A"),
        _report(time=1.0, track_id=2, aircraft_id="B"),
        _report(time=1.0, track_id=1, aircraft_id="C"),
    ]
    output_path = tmp_path / "iff_track_file.csv"
    ReportWriter().write(reports, output_path)
    lines = output_path.read_text(encoding="utf-8").splitlines()[1:]
    aircraft_order = [line.split(",")[2] for line in lines]
    assert aircraft_order == ["C", "B", "A"]  # (1.0,1) < (1.0,2) < (2.0,1)


def test_relative_velocity_formatted_as_semicolon_separated(tmp_path: Path):
    output_path = tmp_path / "iff_track_file.csv"
    ReportWriter().write([_report(relative_velocity=Vector3(1.0, 2.0, 3.0))], output_path)
    line = output_path.read_text(encoding="utf-8").splitlines()[1]
    assert "1.0;2.0;3.0" in line


def test_relative_velocity_none_is_empty_field(tmp_path: Path):
    import csv

    output_path = tmp_path / "iff_track_file.csv"
    ReportWriter().write([_report(relative_velocity=None)], output_path)
    with output_path.open(encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["Relative_Velocity"] == ""


def test_mode_s_address_none_is_empty_field(tmp_path: Path):
    import csv

    output_path = tmp_path / "iff_track_file.csv"
    ReportWriter().write([_report(mode_s_address=None)], output_path)
    with output_path.open(encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["ModeS_Address"] == ""


def test_enum_fields_serialized_as_their_value(tmp_path: Path):
    import csv

    output_path = tmp_path / "iff_track_file.csv"
    ReportWriter().write(
        [_report(mode=IFFMode.MODE5_L2, reply_status=MeasurementStatus.NO_REPLY,
                 friend_foe_status=FriendFoeStatus.SUSPECT, track_status=TrackStatus.CONFIRMED)],
        output_path,
    )
    with output_path.open(encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["Mode"] == "MODE5_L2"
    assert row["Reply_Status"] == "NO_REPLY"
    assert row["Friend_Foe_Status"] == "Suspect"
    assert row["Track_Status"] == "Confirmed"


def test_write_creates_parent_directories(tmp_path: Path):
    output_path = tmp_path / "nested" / "dir" / "iff_track_file.csv"
    ReportWriter().write([_report()], output_path)
    assert output_path.exists()


def test_write_empty_reports_produces_header_only(tmp_path: Path):
    output_path = tmp_path / "iff_track_file.csv"
    ReportWriter().write([], output_path)
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
