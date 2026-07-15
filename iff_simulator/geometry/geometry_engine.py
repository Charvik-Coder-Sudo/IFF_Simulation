"""GeometryEngine: the shared mathematical foundation for every future sensor.

Purpose:
    Implements `GeometryEngine`, which computes a `RelativeState` (Range,
    Azimuth, Elevation, Bearing, Closing Velocity, plus relative
    position/velocity) for a target relative to Ownship, at one
    simulation instant. Every future sensor (IFF, Primary Radar, AESA
    Radar, EO/IR, Sensor Fusion) computes its geometry through this
    module — none of them may re-derive this math independently.

Inputs:
    Plain `Vector3` positions/velocities and a heading float — never a
    `Scenario`, `World`, `Ownship`, or `AircraftState` object directly.

Outputs:
    `RelativeState` instances (one per target, per tick). No CSV,
    plotting, filtering, interrogation, or reply logic happens here.

Coordinate system:
    ENU (East-North-Up): X = East, Y = North, Z = Up, meters, right-handed.

Engineering explanation:
    `GeometryEngine` takes only `Vector3`/`float` primitives as input,
    deliberately with zero dependency on `Scenario`, `World`, `Ownship`,
    or `AircraftState`. This keeps it a pure, standalone mathematical
    layer: any future sensor (including ones with no concept of a
    "Scenario" at all, e.g. a standalone EO/IR bench-test harness) can
    call it directly. The caller (a future sensor module) is responsible
    for extracting `ownship.position` / `ownship.velocity` /
    `ownship.heading` and each target's `AircraftState` fields and
    passing them in explicitly.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from ..domain import Vector3
from .relative_state import RelativeState
from .vector_math import safe_normalize


class GeometryEngine:
    """Computes RelativeState geometry between Ownship and targets.

    Purpose:
        The single, reusable place where "relative position, relative
        velocity, range, azimuth, elevation, bearing, and closing
        velocity" are derived from raw kinematics. No other module may
        duplicate this math.

    Inputs:
        See `compute_relative_state` / `compute_batch`.

    Outputs:
        `RelativeState` instances.

    Engineering explanation:
        Stateless by design (every method is a pure function of its
        arguments) — a `GeometryEngine` instance holds no per-call
        state, so a single instance (or, equivalently, the class itself
        via its `@staticmethod`s) can be safely shared and called
        concurrently by multiple future sensor modules.
    """

    @staticmethod
    def compute_relative_state(
        target_id: str,
        time: float,
        ownship_position: Vector3,
        ownship_velocity: Vector3,
        ownship_heading_deg: float,
        target_position: Vector3,
        target_velocity: Vector3,
    ) -> RelativeState:
        """Compute one target's RelativeState relative to Ownship, at one instant.

        Purpose:
            The reference (scalar, `math`-module) implementation of the
            full Range/Azimuth/Elevation/Bearing/Closing-Velocity
            geometry for a single target.

        Inputs:
            target_id: identifier of the target.
            time: simulation time this state corresponds to.
            ownship_position, ownship_velocity: Ownship's Vector3
                position (meters) and velocity (m/s), ENU.
            ownship_heading_deg: Ownship's heading, degrees, compass
                convention (0 = North, 90 = East).
            target_position, target_velocity: the target's Vector3
                position (meters) and velocity (m/s), ENU.

        Outputs:
            A `RelativeState` for this (Ownship, target, time) triple.

        Units:
            meters and meters/second in; meters, meters/second, and
            degrees out (see `RelativeState` field docs for exact units
            per field).

        Mathematics:
            R = Pt - Po                          (relative position)
            Vr = Vt - Vo                          (relative velocity)
            r = |R| = sqrt(dx^2 + dy^2 + dz^2)    (range)
            azimuth = atan2(dy, dx)                (degrees)
            elevation = asin(dz / r)                (degrees)
            true_bearing = heading(R)              (degrees, compass)
            bearing = (true_bearing - ownship_heading_deg) mod 360
            closing_velocity = -(Vr . R_hat)       (R_hat = R / r)

        Engineering reasoning:
            Closing velocity is the *negative* of the raw range-rate
            (Vr . R_hat): R points from Ownship to the target, so a
            target flying back down that line toward Ownship has a
            relative velocity component *opposite* R (a negative
            Vr . R_hat), yet must report a *positive* closing velocity
            (the engineering/TCAS convention: positive = approaching).
            Negating the raw range-rate gives exactly that convention.
            When range is exactly 0 (Ownship and target coincide),
            azimuth/elevation/bearing/closing-velocity are all
            direction-dependent quantities with no defined direction to
            report; they default to 0.0 rather than raising or
            producing NaN, per this phase's edge-case requirements.
        """
        relative_position = target_position - ownship_position
        relative_velocity = target_velocity - ownship_velocity

        range_m = relative_position.magnitude()

        if range_m == 0.0:
            azimuth_deg = 0.0
            elevation_deg = 0.0
            bearing_deg = 0.0
            closing_velocity_mps = 0.0
        else:
            dx, dy, dz = relative_position.x, relative_position.y, relative_position.z

            azimuth_deg = math.degrees(math.atan2(dy, dx))

            sin_elevation = max(-1.0, min(1.0, dz / range_m))
            elevation_deg = math.degrees(math.asin(sin_elevation))

            true_bearing_deg = relative_position.heading()
            bearing_deg = (true_bearing_deg - ownship_heading_deg) % 360.0

            line_of_sight = safe_normalize(relative_position)
            closing_velocity_mps = -relative_velocity.dot(line_of_sight)

        return RelativeState(
            target_id=target_id,
            time=time,
            relative_position=relative_position,
            relative_velocity=relative_velocity,
            range_m=range_m,
            azimuth_deg=azimuth_deg,
            elevation_deg=elevation_deg,
            bearing_deg=bearing_deg,
            closing_velocity_mps=closing_velocity_mps,
        )

    @classmethod
    def compute_batch(
        cls,
        time: float,
        ownship_position: Vector3,
        ownship_velocity: Vector3,
        ownship_heading_deg: float,
        targets: Iterable[tuple[str, Vector3, Vector3]],
    ) -> list[RelativeState]:
        """Compute RelativeState for many targets against one Ownship state.

        Purpose:
            A numpy-vectorized bulk counterpart to
            `compute_relative_state`, for the common case of computing
            every target's geometry in one simulation tick at once —
            the shape a future AESA radar (many simultaneous beams/
            targets) or Sensor Fusion (many tracks) module needs.

        Inputs:
            time, ownship_position, ownship_velocity, ownship_heading_deg:
                same as `compute_relative_state`.
            targets: iterable of (target_id, target_position, target_velocity)
                tuples, one per target.

        Outputs:
            List of `RelativeState`, one per input target, in input order.

        Units:
            Same as `compute_relative_state`.

        Mathematics:
            Identical formulas to `compute_relative_state`, applied
            elementwise across all N targets via numpy array operations
            instead of a Python-level loop.

        Engineering reasoning:
            `math.atan2`/`math.asin` and `numpy.arctan2`/`numpy.arcsin`
            both call the platform's correctly-rounded libm
            implementation for a given double input, so this method is
            numerically consistent with `compute_relative_state` for
            the same inputs (verified by a dedicated cross-check test),
            while scaling to many targets in one vectorized pass instead
            of N Python-level function calls. Division-by-zero at
            Range == 0 is avoided with an explicit boolean mask, not a
            try/except or a NaN-then-clean-up pass, so no NaN is ever
            materialized.
        """
        targets = list(targets)
        if not targets:
            return []

        target_ids = [target_id for target_id, _, _ in targets]
        target_positions = np.array(
            [[position.x, position.y, position.z] for _, position, _ in targets],
            dtype=np.float64,
        )
        target_velocities = np.array(
            [[velocity.x, velocity.y, velocity.z] for _, _, velocity in targets],
            dtype=np.float64,
        )

        ownship_pos = np.array([ownship_position.x, ownship_position.y, ownship_position.z])
        ownship_vel = np.array([ownship_velocity.x, ownship_velocity.y, ownship_velocity.z])

        relative_positions = target_positions - ownship_pos
        relative_velocities = target_velocities - ownship_vel

        dx = relative_positions[:, 0]
        dy = relative_positions[:, 1]
        dz = relative_positions[:, 2]

        ranges = np.sqrt(dx**2 + dy**2 + dz**2)
        nonzero = ranges > 0.0
        safe_ranges = np.where(nonzero, ranges, 1.0)  # avoid 0/0; result discarded where masked out

        azimuths = np.zeros_like(ranges)
        azimuths[nonzero] = np.degrees(np.arctan2(dy[nonzero], dx[nonzero]))

        sin_elevation = np.clip(dz / safe_ranges, -1.0, 1.0)
        elevations = np.zeros_like(ranges)
        elevations[nonzero] = np.degrees(np.arcsin(sin_elevation[nonzero]))

        true_bearings = np.zeros_like(ranges)
        true_bearings[nonzero] = np.degrees(np.arctan2(dx[nonzero], dy[nonzero])) % 360.0
        bearings = np.where(nonzero, (true_bearings - ownship_heading_deg) % 360.0, 0.0)

        line_of_sight = np.zeros_like(relative_positions)
        line_of_sight[nonzero] = relative_positions[nonzero] / ranges[nonzero, None]
        closing_velocities = np.zeros_like(ranges)
        closing_velocities[nonzero] = -np.einsum(
            "ij,ij->i", relative_velocities[nonzero], line_of_sight[nonzero]
        )

        results: list[RelativeState] = []
        for i, target_id in enumerate(target_ids):
            results.append(
                RelativeState(
                    target_id=target_id,
                    time=time,
                    relative_position=Vector3(
                        float(relative_positions[i, 0]),
                        float(relative_positions[i, 1]),
                        float(relative_positions[i, 2]),
                    ),
                    relative_velocity=Vector3(
                        float(relative_velocities[i, 0]),
                        float(relative_velocities[i, 1]),
                        float(relative_velocities[i, 2]),
                    ),
                    range_m=float(ranges[i]),
                    azimuth_deg=float(azimuths[i]),
                    elevation_deg=float(elevations[i]),
                    bearing_deg=float(bearings[i]),
                    closing_velocity_mps=float(closing_velocities[i]),
                )
            )
        return results
