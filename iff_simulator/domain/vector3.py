"""Immutable 3D vector primitive used throughout the domain model.

Purpose:
    Provide a single, reusable, immutable 3D vector type for positions,
    velocities, and accelerations, replacing ad-hoc (x, y, z) float
    triples or DataFrame columns with a proper value type.

Inputs:
    Three scalar components (x, y, z) at construction time.

Outputs:
    A `Vector3` instance and the arithmetic/geometric operations defined
    on it (+, -, *, /, magnitude, normalize, dot, cross, distance,
    heading).

Engineering explanation:
    Vector3 is deliberately a pure math primitive with no knowledge of
    sensors, aircraft, or scenarios — it belongs equally to every future
    module (Ownship, Geometry, PSR, IFF) that needs 3D vector algebra.
    It is frozen (immutable) so it can be freely shared/copied between
    domain objects without defensive copying or aliasing bugs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Vector3:
    """An immutable 3D vector.

    Purpose:
        Represent a 3D quantity (position, velocity, or acceleration) as
        a single value type with well-defined arithmetic and geometric
        operations, instead of three loose float fields.

    Inputs:
        x, y, z: float components of the vector.

    Outputs:
        A `Vector3` value, or the scalar/vector result of an operation
        performed on it.

    Engineering explanation:
        All operations return new `Vector3` instances (or plain floats
        for scalar results) rather than mutating in place, matching the
        "frozen" nature of the type and avoiding shared-mutable-state
        bugs when the same vector is referenced from multiple places.
    """

    x: float
    y: float
    z: float

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vector3":
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> "Vector3":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Vector3":
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)

    def magnitude(self) -> float:
        """Return the Euclidean length of the vector."""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self) -> "Vector3":
        """Return a unit-length vector in the same direction.

        Raises:
            ValueError: if the vector has zero magnitude.
        """
        length = self.magnitude()
        if length == 0.0:
            raise ValueError("Cannot normalize a zero-length vector.")
        return self / length

    def dot(self, other: "Vector3") -> float:
        """Return the scalar dot product with another vector."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vector3") -> "Vector3":
        """Return the vector cross product with another vector."""
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def distance_to(self, other: "Vector3") -> float:
        """Return the Euclidean distance to another vector (treated as a point)."""
        return (self - other).magnitude()

    def heading(self) -> float:
        """Return the compass-style heading of this vector in the XY plane.

        Returns:
            Degrees in [0, 360), measured clockwise from the +Y axis
            (i.e. 0 deg = +Y, 90 deg = +X). This is a generic vector
            convention local to `Vector3` and is independent of any
            sensor-specific azimuth convention used elsewhere.
        """
        degrees = math.degrees(math.atan2(self.x, self.y))
        return degrees % 360.0
