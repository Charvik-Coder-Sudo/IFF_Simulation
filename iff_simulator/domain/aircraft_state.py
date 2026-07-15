"""Live, time-varying kinematic state of a single aircraft at one instant.

Purpose:
    Represent everything about an aircraft that changes from sample to
    sample: its kinematics at a specific time. `Aircraft` owns identity
    and configuration; `AircraftState` owns the numbers that move.

Inputs:
    Parsed by `GroundTruthLoader` from one row of a `.tdf` recording.

Outputs:
    A mutable `AircraftState` instance, held in a `Scenario`'s recorded
    state history and referenced as the "current" state by `World`
    (and, for the Ownship aircraft, read live through `Ownship`).

Engineering explanation:
    The recorded `.tdf` data already includes a sensor-derived polar
    measurement (slant range, azimuth, elevation) alongside the
    Cartesian position. These are preserved verbatim as extra fields
    rather than recomputed from `position`, because recomputing them
    (e.g. range = position.magnitude()) reintroduces floating-point
    rounding that would silently diverge from the originally recorded
    values and break byte-for-byte CSV reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .vector3 import Vector3


@dataclass(slots=True)
class AircraftState:
    """Mutable kinematic state of one aircraft at one point in time.

    Purpose:
        Hold the position, velocity, acceleration, heading, and
        liveness of an aircraft at a single instant, plus the raw
        recorded sensor measurement (range/azimuth/elevation) for that
        same instant.

    Inputs:
        time: sample time stamp.
        position, velocity, acceleration: `Vector3` values.
        heading: compass-style heading, degrees.
        alive: whether the aircraft is present/valid at this sample.
        range_m, azimuth_deg, elevation_deg: the raw recorded polar
            measurement, preserved verbatim from the source `.tdf` file.

    Outputs:
        Consumed by `Scenario`, `GroundTruthInspector`,
        `GroundTruthStatistics`, and `TrajectoryPlotter`.

    Engineering explanation:
        Declared mutable (not frozen) because `World`/`SimulationClock`
        are expected to advance which `AircraftState` is "current" as
        simulated time progresses in later phases; Phase 1 itself never
        mutates a state after construction.
    """

    time: float
    position: Vector3
    velocity: Vector3
    acceleration: Vector3 = field(default_factory=lambda: Vector3(0.0, 0.0, 0.0))
    heading: float = 0.0
    alive: bool = True
    range_m: float = 0.0
    azimuth_deg: float = 0.0
    elevation_deg: float = 0.0
