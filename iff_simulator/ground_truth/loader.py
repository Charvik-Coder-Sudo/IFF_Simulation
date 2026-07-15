"""Discovers and loads raw aircraft trajectory recordings (.tdf files).

Purpose:
    Implements `GroundTruthLoader`, the single entry point that turns a
    folder of `.tdf` recordings into in-memory pandas DataFrames.

Inputs:
    A directory path (the Aircrafts folder) containing one or more
    `*.tdf` files. Files are discovered automatically — never hardcoded.

Outputs:
    `dict[str, pandas.DataFrame]` mapping TargetID -> raw trajectory
    DataFrame, with columns exactly matching
    `iff_simulator.ground_truth.models.REQUIRED_COLUMNS`.

Engineering explanation:
    Each `.tdf` file begins with a one-line target header (e.g.
    "TARGET 1 ,") followed by a CSV header row and CSV data rows. The
    trailing "TgtId" name in that CSV header has no corresponding data
    field in any row (a quirk of the recorder), so it is dropped and
    replaced with an explicit TargetID column parsed from the file's own
    header line. This guarantees every sample is traceable to its source
    aircraft regardless of how files are later merged or reordered.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .models import REQUIRED_COLUMNS

_TARGET_HEADER_PATTERN = re.compile(r"TARGET\s+(\d+)", re.IGNORECASE)

_RAW_VELOCITY_RENAME = {"Vx": "VX", "Vy": "VY", "Vz": "VZ"}


class GroundTruthLoader:
    """Scans the Aircrafts folder and loads every `.tdf` recording.

    Purpose:
        Provide a single, reusable component that turns raw `.tdf` files
        on disk into clean, consistently-shaped pandas DataFrames, ready
        for validation and merging.

    Inputs:
        `aircrafts_dir`: path to the folder holding `.tdf` files.

    Outputs:
        `load()` returns `dict[str, pandas.DataFrame]` keyed by TargetID.

    Engineering explanation:
        Uses `pathlib.Path.glob("*.tdf")` so any number of aircraft files
        can be added or removed from the Aircrafts folder without any
        code change. Column cleanup (stripping whitespace, dropping the
        empty TgtId column, renaming Vx/Vy/Vz to VX/VY/VZ) is centralized
        here so every downstream module can rely on
        `REQUIRED_COLUMNS` being present and correctly named.
    """

    def __init__(self, aircrafts_dir: Path | str) -> None:
        self.aircrafts_dir = Path(aircrafts_dir)
        if not self.aircrafts_dir.is_dir():
            raise FileNotFoundError(
                f"Aircrafts directory not found: {self.aircrafts_dir}"
            )

    def discover_files(self) -> list[Path]:
        """Return every `.tdf` file in the aircrafts directory, sorted by name."""
        files = sorted(self.aircrafts_dir.glob("*.tdf"))
        if not files:
            raise FileNotFoundError(
                f"No .tdf files found in {self.aircrafts_dir}"
            )
        return files

    def _extract_target_id(self, file_path: Path) -> str:
        """Parse the TargetID from the first line of a .tdf file."""
        with file_path.open("r", encoding="utf-8") as handle:
            first_line = handle.readline()
        match = _TARGET_HEADER_PATTERN.search(first_line)
        if not match:
            raise ValueError(
                f"Could not parse target header from {file_path.name}: "
                f"'{first_line.strip()}'"
            )
        return f"TARGET_{match.group(1)}"

    def _load_file(self, file_path: Path) -> pd.DataFrame:
        """Load a single .tdf file into a clean, canonically-shaped DataFrame."""
        target_id = self._extract_target_id(file_path)

        df = pd.read_csv(file_path, skiprows=1)
        df.columns = [str(column).strip() for column in df.columns]
        df = df.rename(columns=_RAW_VELOCITY_RENAME)

        if "TgtId" in df.columns:
            df = df.drop(columns=["TgtId"])

        df.insert(1, "TargetID", target_id)

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"{file_path.name} is missing required columns: {missing}"
            )
        return df[REQUIRED_COLUMNS].reset_index(drop=True)

    def load(self) -> dict[str, pd.DataFrame]:
        """Scan the Aircrafts folder and load every trajectory into memory.

        Returns:
            dict mapping TargetID -> raw trajectory DataFrame.
        """
        trajectories: dict[str, pd.DataFrame] = {}
        for file_path in self.discover_files():
            df = self._load_file(file_path)
            target_id = df["TargetID"].iloc[0]
            if target_id in trajectories:
                raise ValueError(
                    f"Duplicate TargetID '{target_id}' encountered in "
                    f"{file_path.name}; each aircraft must be unique."
                )
            trajectories[target_id] = df
        return trajectories
