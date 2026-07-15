"""Discovers and loads raw aircraft trajectory recordings (.tdf files).

Purpose:
    Implements `GroundTruthLoader`, the single entry point that turns a
    folder of `.tdf` recordings into a domain-object `Scenario`.

Inputs:
    A directory path (the Aircrafts folder) containing one or more
    `*.tdf` files. Files are discovered automatically — never hardcoded.

Outputs:
    A `Scenario` containing one `Aircraft` and one time-ordered list of
    `AircraftState` per discovered file.

Engineering explanation:
    Each `.tdf` file begins with a one-line target header (e.g.
    "TARGET 1 ,") followed by a CSV header row and CSV data rows. The
    trailing "TgtId" name in that CSV header has no corresponding data
    field in any row (a quirk of the recorder), so it is dropped and
    replaced with an explicit aircraft ID parsed from the file's own
    header line. Pandas is used here purely as an implementation detail
    for parsing the CSV body — nothing downstream of this module ever
    sees a DataFrame; every other runtime module operates on the
    `Scenario` / `Aircraft` / `AircraftState` domain objects instead.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ..domain import Aircraft, AircraftState, Scenario, Vector3

_TARGET_HEADER_PATTERN = re.compile(r"TARGET\s+(\d+)", re.IGNORECASE)

_RAW_VELOCITY_RENAME = {"Vx": "VX", "Vy": "VY", "Vz": "VZ"}

#: Column order expected in every cleaned per-file DataFrame, before it
#: is converted into `AircraftState` objects.
_RAW_COLUMNS = [
    "Time",
    "X",
    "Y",
    "Z",
    "VX",
    "VY",
    "VZ",
    "Range",
    "Azimuth",
    "Elevation",
]


class GroundTruthLoader:
    """Scans the Aircrafts folder and loads every `.tdf` recording.

    Purpose:
        Provide a single, reusable component that turns raw `.tdf` files
        on disk into a `Scenario` of domain objects, ready for
        validation, merging, and inspection.

    Inputs:
        `aircrafts_dir`: path to the folder holding `.tdf` files.

    Outputs:
        `load()` returns a `Scenario`.

    Engineering explanation:
        Uses `pathlib.Path.glob("*.tdf")` so any number of aircraft files
        can be added or removed from the Aircrafts folder without any
        code change. Column cleanup (stripping whitespace, dropping the
        empty TgtId column, renaming Vx/Vy/Vz to VX/VY/VZ) happens on an
        internal, throwaway DataFrame per file; the returned `Scenario`
        exposes only domain objects.
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

    def _extract_aircraft_id(self, file_path: Path) -> str:
        """Parse the aircraft ID from the first line of a .tdf file."""
        with file_path.open("r", encoding="utf-8") as handle:
            first_line = handle.readline()
        match = _TARGET_HEADER_PATTERN.search(first_line)
        if not match:
            raise ValueError(
                f"Could not parse target header from {file_path.name}: "
                f"'{first_line.strip()}'"
            )
        return f"TARGET_{match.group(1)}"

    def _read_raw_frame(self, file_path: Path) -> pd.DataFrame:
        """Read one .tdf file's CSV body into a cleaned, throwaway DataFrame.

        This is the only place pandas is used to represent trajectory
        data; the DataFrame never leaves this method.
        """
        df = pd.read_csv(file_path, skiprows=1)
        df.columns = [str(column).strip() for column in df.columns]
        df = df.rename(columns=_RAW_VELOCITY_RENAME)

        if "TgtId" in df.columns:
            df = df.drop(columns=["TgtId"])

        missing = [c for c in _RAW_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"{file_path.name} is missing required columns: {missing}"
            )
        return df[_RAW_COLUMNS].reset_index(drop=True)

    def _build_states(self, df: pd.DataFrame) -> list[AircraftState]:
        """Convert a cleaned raw DataFrame into a time-ordered list of AircraftState."""
        states: list[AircraftState] = []
        for row in df.itertuples(index=False):
            position = Vector3(float(row.X), float(row.Y), float(row.Z))
            velocity = Vector3(float(row.VX), float(row.VY), float(row.VZ))
            states.append(
                AircraftState(
                    time=row.Time,
                    position=position,
                    velocity=velocity,
                    heading=velocity.heading(),
                    range_m=float(row.Range),
                    azimuth_deg=float(row.Azimuth),
                    elevation_deg=float(row.Elevation),
                )
            )
        return states

    def _load_file(self, file_path: Path) -> tuple[Aircraft, list[AircraftState]]:
        """Load a single .tdf file into an (Aircraft, state history) pair."""
        aircraft_id = self._extract_aircraft_id(file_path)
        raw_frame = self._read_raw_frame(file_path)
        states = self._build_states(raw_frame)
        aircraft = Aircraft(aircraft_id=aircraft_id)
        return aircraft, states

    def load(self) -> Scenario:
        """Scan the Aircrafts folder and load every trajectory into a Scenario.

        Returns:
            A `Scenario` with one `Aircraft` and one recorded state
            history per discovered `.tdf` file.
        """
        aircraft_list: list[Aircraft] = []
        state_history: dict[str, list[AircraftState]] = {}

        for file_path in self.discover_files():
            aircraft, states = self._load_file(file_path)
            if aircraft.aircraft_id in state_history:
                raise ValueError(
                    f"Duplicate aircraft_id '{aircraft.aircraft_id}' encountered "
                    f"in {file_path.name}; each aircraft must be unique."
                )
            aircraft_list.append(aircraft)
            state_history[aircraft.aircraft_id] = states

        return Scenario(aircraft_list, state_history)
