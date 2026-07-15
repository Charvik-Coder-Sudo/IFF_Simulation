"""Data model definitions for the Ground Truth subsystem.

Purpose:
    Defines the canonical schema for a single aircraft kinematic sample.
    Every future module (Ownship, Geometry, Airborne PSR, IFF, Scheduler,
    Receiver, Decoder) is expected to consume ground-truth data shaped
    exactly like `GroundTruthSample` / `REQUIRED_COLUMNS`, so this file is
    the single source of truth for the data contract.

Inputs:
    None directly — this module only defines types and constants.

Outputs:
    `GroundTruthSample` dataclass and `REQUIRED_COLUMNS` column-order
    constant, imported by every other ground_truth module.

Engineering explanation:
    Position (X, Y, Z) is expressed in a local Cartesian frame (meters),
    velocity (VX, VY, VZ) in meters/second, and Range/Azimuth/Elevation
    are the spherical representation of the same position as measured
    from the sensor origin. Recording both representations avoids
    repeated geometric conversions in later phases (which are explicitly
    out of scope for Phase 1).
"""

from __future__ import annotations

from dataclasses import dataclass

#: Canonical column names, in canonical order, for one ground-truth sample.
#: This exact order is used for the merged ground_truth.csv output.
REQUIRED_COLUMNS: list[str] = [
    "Time",
    "TargetID",
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


@dataclass(frozen=True, slots=True)
class GroundTruthSample:
    """One kinematic sample of a single aircraft at a single instant in time.

    Purpose:
        Represents one row of recorded ground-truth trajectory data for a
        single aircraft. This is the atomic, documented unit of data that
        every downstream subsystem is built on top of.

    Inputs:
        Constructed from one row of a validated, merged ground-truth
        DataFrame (see `GroundTruthMerger` / `GroundTruthInspector`).

    Outputs:
        Consumed anywhere a single, strongly-typed aircraft state is
        needed instead of a raw DataFrame row.

    Engineering explanation:
        Kept as a frozen (immutable) dataclass with `slots=True` so
        instances are lightweight and cannot be mutated accidentally once
        created, matching the read-only nature of recorded ground truth.
    """

    time: float
    """Sample time stamp. Monotonically increasing per target, constant
    timestep within a target's series."""

    target_id: str
    """Unique identifier of the aircraft that produced this sample,
    e.g. "TARGET_1". Parsed from the source .tdf file's own header line."""

    x: float
    """X position component, meters, local Cartesian sensor-centered frame."""

    y: float
    """Y position component, meters, local Cartesian sensor-centered frame."""

    z: float
    """Z position component, meters, local Cartesian sensor-centered frame
    (vertical/altitude-like axis)."""

    vx: float
    """X velocity component, meters/second."""

    vy: float
    """Y velocity component, meters/second."""

    vz: float
    """Z velocity component, meters/second."""

    range_m: float
    """Slant range from the sensor origin to the aircraft, meters."""

    azimuth_deg: float
    """Azimuth angle from the sensor origin to the aircraft, degrees."""

    elevation_deg: float
    """Elevation angle from the sensor origin to the aircraft, degrees."""

    @classmethod
    def from_row(cls, row: dict) -> "GroundTruthSample":
        """Build a `GroundTruthSample` from a dict-like row keyed by
        `REQUIRED_COLUMNS` (e.g. a pandas `Series` from a ground-truth
        DataFrame)."""
        return cls(
            time=float(row["Time"]),
            target_id=str(row["TargetID"]),
            x=float(row["X"]),
            y=float(row["Y"]),
            z=float(row["Z"]),
            vx=float(row["VX"]),
            vy=float(row["VY"]),
            vz=float(row["VZ"]),
            range_m=float(row["Range"]),
            azimuth_deg=float(row["Azimuth"]),
            elevation_deg=float(row["Elevation"]),
        )

    def to_row(self) -> dict:
        """Convert back to a dict keyed by `REQUIRED_COLUMNS`."""
        return {
            "Time": self.time,
            "TargetID": self.target_id,
            "X": self.x,
            "Y": self.y,
            "Z": self.z,
            "VX": self.vx,
            "VY": self.vy,
            "VZ": self.vz,
            "Range": self.range_m,
            "Azimuth": self.azimuth_deg,
            "Elevation": self.elevation_deg,
        }
