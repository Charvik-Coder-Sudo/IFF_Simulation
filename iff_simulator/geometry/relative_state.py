"""Immutable snapshot of one target's kinematic relationship to Ownship.

Purpose:
    Defines `RelativeState`, the output type of `GeometryEngine`: every
    geometric quantity one target has relative to Ownship, at one
    simulation time. This is the shared data contract every future
    sensor (IFF, PSR, AESA, EO/IR, Sensor Fusion) will consume instead
    of recomputing geometry itself.

Coordinate system:
    ENU (East-North-Up): X = East, Y = North, Z = Up, meters. All
    position/velocity fields here are expressed in this frame.

Engineering explanation:
    Frozen (immutable) for the same reason `AircraftState`'s kinematic
    values and `Vector3` are: a `RelativeState` represents a computed
    fact about one instant and must never be mutated after creation, so
    it can be freely shared/cached/passed to multiple consumers (e.g.
    IFF interrogation logic and a fusion tracker reading the same tick's
    geometry) without aliasing risk.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..domain import Vector3


@dataclass(frozen=True, slots=True)
class RelativeState:
    """Geometric relationship of one target to Ownship, at one instant.

    Purpose:
        Carry every quantity `GeometryEngine` computes for a single
        (Ownship, target, time) triple: relative kinematics plus the
        derived range/azimuth/elevation/bearing/closing-velocity.

    Inputs:
        Constructed exclusively by `GeometryEngine`; not intended to be
        hand-built by callers.

    Outputs:
        Consumed by any future sensor module that needs "where is this
        target relative to me" without recomputing the geometry itself.

    Engineering explanation:
        Azimuth and Bearing are two different, deliberately distinct
        angles: Azimuth is the sensor/math-convention angle in the
        Ownship-centered ENU plane (atan2(dy, dx): 0 deg = +X/East,
        90 deg = +Y/North), as radar/antenna systems reference it.
        Bearing is the navigation-convention angle relative to
        Ownship's own heading (0 deg = dead ahead, 90 deg = right,
        wrapped to [0, 360)) — what a gimbaled sensor or pilot display
        needs to know "which way to look" relative to the platform's
        nose. Neither is a substitute for the other.
    """

    target_id: str
    """Identifier of the target this RelativeState describes (matches
    the corresponding Scenario aircraft_id)."""

    time: float
    """Simulation time this RelativeState was computed for."""

    relative_position: Vector3
    """Pt - Po: target position minus Ownship position, meters, ENU."""

    relative_velocity: Vector3
    """Vt - Vo: target velocity minus Ownship velocity, meters/second, ENU."""

    range_m: float
    """Straight-line (slant) distance between Ownship and target, meters.
    Always >= 0; exactly 0.0 when Ownship and target coincide."""

    azimuth_deg: float
    """Sensor/math-convention angle to the target in the Ownship-centered
    ENU horizontal plane: atan2(dy, dx), degrees, range (-180, 180].
    0 deg = +X (East), 90 deg = +Y (North). 0.0 by convention when
    range_m == 0 (direction undefined)."""

    elevation_deg: float
    """Angle above/below the ENU horizontal (X-Y) plane: asin(dz / range),
    degrees, range [-90, 90]. Positive = above Ownship, negative = below.
    0.0 by convention when range_m == 0 (direction undefined)."""

    bearing_deg: float
    """True bearing to the target minus Ownship's own heading, wrapped to
    [0, 360): 0 deg = dead ahead of Ownship, 90 deg = directly to its
    right, 180 deg = directly behind, 270 deg = directly to its left.
    0.0 by convention when range_m == 0 (direction undefined)."""

    closing_velocity_mps: float
    """Rate at which range is shrinking, meters/second. Positive means
    the target is approaching (range decreasing); negative means it is
    receding (range increasing). 0.0 by convention when range_m == 0."""
