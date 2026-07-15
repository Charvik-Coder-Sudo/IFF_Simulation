"""Shared pytest fixtures for the Ground Truth subsystem tests."""

from __future__ import annotations

from pathlib import Path

import pytest

_TDF_HEADER = "Time,X,Y,Z,Vx,Vy,Vz,Range,Azimuth,Elevation,TgtId \n"


def _write_tdf(path: Path, target_number: int, rows: list[tuple]) -> None:
    lines = [f"TARGET {target_number} ,\n", _TDF_HEADER]
    for row in rows:
        lines.append(",".join(str(v) for v in row) + "\n")
    path.write_text("".join(lines), encoding="utf-8")


def _make_rows(n: int, start_time: int = 1) -> list[tuple]:
    rows = []
    for i in range(n):
        t = start_time + i
        rows.append(
            (
                t,
                float(i),
                float(i) * 2,
                100.0,
                10.0,
                5.0,
                0.0,
                float(i) * 3,
                45.0,
                0.0,
            )
        )
    return rows


@pytest.fixture
def aircrafts_dir(tmp_path: Path) -> Path:
    """A temporary Aircrafts folder with two small, well-formed .tdf targets."""
    folder = tmp_path / "Aircrafts"
    folder.mkdir()
    _write_tdf(folder / "Target_1_fixture.tdf", 1, _make_rows(5))
    _write_tdf(folder / "Target_2_fixture.tdf", 2, _make_rows(4))
    return folder
