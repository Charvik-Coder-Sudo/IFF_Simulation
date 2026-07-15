"""Reusable vector-math utilities, built strictly on top of `Vector3`.

Purpose:
    Provide the small set of vector operations every geometry
    computation needs (distance, magnitude, normalize, dot, cross,
    angle_between, heading, bearing) as plain, independently-testable
    functions, so `GeometryEngine` (and any future sensor module) never
    has to re-derive this math itself.

Coordinate system:
    ENU (East-North-Up): X = East, Y = North, Z = Up, meters. Every
    function in this module assumes its `Vector3` arguments are already
    expressed in this frame.

Engineering explanation:
    Every function that already exists as a `Vector3` method
    (`distance`, `magnitude`, `normalize`, `dot`, `cross`, `heading`) is
    a thin one-line delegate to that method — there is exactly one
    implementation of each of those operations in the codebase, here or
    in `Vector3`, never both. Only `angle_between`, `bearing`, and
    `safe_normalize` are genuinely new: they do not exist on `Vector3`
    at all, so they are implemented (not duplicated) here.
"""

from __future__ import annotations

import math

from ..domain import Vector3

#: The zero vector, returned by `safe_normalize` for zero-length input
#: instead of raising, since a coincident line-of-sight is an expected,
#: valid geometry input (e.g. Range == 0), not an error condition.
ZERO_VECTOR = Vector3(0.0, 0.0, 0.0)


def distance(a: Vector3, b: Vector3) -> float:
    """Euclidean distance between two points.

    Purpose:
        Reusable point-to-point distance, delegating to `Vector3.distance_to`.
    Inputs:
        a, b: `Vector3` positions, meters.
    Outputs:
        float, meters, always >= 0.
    Units:
        meters in, meters out.
    Mathematics:
        |a - b| = sqrt((ax-bx)^2 + (ay-by)^2 + (az-bz)^2)
    Engineering reasoning:
        Delegates entirely to `Vector3.distance_to` so there is a single
        implementation of Euclidean distance in the codebase.
    """
    return a.distance_to(b)


def magnitude(v: Vector3) -> float:
    """Euclidean length of a vector.

    Purpose:
        Reusable vector length, delegating to `Vector3.magnitude`.
    Inputs:
        v: a `Vector3`.
    Outputs:
        float, always >= 0, same units as `v`'s components.
    Units:
        matches `v` (meters for position/relative-position, m/s for velocity).
    Mathematics:
        |v| = sqrt(vx^2 + vy^2 + vz^2)
    Engineering reasoning:
        Delegates entirely to `Vector3.magnitude`.
    """
    return v.magnitude()


def normalize(v: Vector3) -> Vector3:
    """Unit vector in the direction of `v`.

    Purpose:
        Reusable normalization, delegating to `Vector3.normalize`.
    Inputs:
        v: a non-zero `Vector3`.
    Outputs:
        `Vector3` with magnitude 1.0, same direction as `v`.
    Units:
        dimensionless (unit vector).
    Mathematics:
        v_hat = v / |v|
    Engineering reasoning:
        Delegates entirely to `Vector3.normalize`, which raises
        `ValueError` on a zero-length vector by design. Use
        `safe_normalize` instead when a zero-length input is a valid,
        expected case (e.g. Range == 0) rather than a programming error.
    """
    return v.normalize()


def safe_normalize(v: Vector3) -> Vector3:
    """Unit vector in the direction of `v`, or the zero vector if `v` is zero-length.

    Purpose:
        A total (never-raising) variant of `normalize`, for geometry
        code where a zero-length vector (e.g. a target coincident with
        Ownship) is an expected input, not an error.
    Inputs:
        v: any `Vector3`, including the zero vector.
    Outputs:
        `Vector3` with magnitude 1.0 in the direction of `v`, or
        `ZERO_VECTOR` if `v` has zero magnitude.
    Units:
        dimensionless (unit vector) or zero vector.
    Mathematics:
        v_hat = v / |v| if |v| > 0, else (0, 0, 0)
    Engineering reasoning:
        `Vector3.normalize` deliberately raises on a zero-length vector,
        which is the right contract for `Vector3` itself. Geometry
        computations over live simulation data must not crash just
        because a target momentarily coincides with Ownship, so this
        function exists specifically to make that case a defined,
        NaN-free, exception-free result instead.
    """
    length = v.magnitude()
    if length == 0.0:
        return ZERO_VECTOR
    return v / length


def dot(a: Vector3, b: Vector3) -> float:
    """Scalar (dot) product of two vectors.

    Purpose:
        Reusable dot product, delegating to `Vector3.dot`.
    Inputs:
        a, b: `Vector3` values.
    Outputs:
        float scalar.
    Units:
        product of `a` and `b`'s units (e.g. m^2/s for a position dotted
        with a velocity).
    Mathematics:
        a . b = ax*bx + ay*by + az*bz
    Engineering reasoning:
        Delegates entirely to `Vector3.dot`.
    """
    return a.dot(b)


def cross(a: Vector3, b: Vector3) -> Vector3:
    """Vector (cross) product of two vectors.

    Purpose:
        Reusable cross product, delegating to `Vector3.cross`.
    Inputs:
        a, b: `Vector3` values.
    Outputs:
        `Vector3` perpendicular to both `a` and `b`.
    Units:
        product of `a` and `b`'s units.
    Mathematics:
        a x b = (ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx)
    Engineering reasoning:
        Delegates entirely to `Vector3.cross`.
    """
    return a.cross(b)


def angle_between(a: Vector3, b: Vector3) -> float:
    """Angle between two vectors, degrees, in [0, 180].

    Purpose:
        Reusable angular separation between any two vectors — new
        functionality; `Vector3` has no equivalent method.
    Inputs:
        a, b: `Vector3` values, either may be the zero vector.
    Outputs:
        float degrees in [0, 180]. Returns 0.0 if either input has zero
        magnitude (angle is undefined for a zero-length vector, and 0.0
        is a safe, NaN-free default).
    Units:
        degrees out; inputs may be any consistent unit (the angle does
        not depend on either vector's magnitude).
    Mathematics:
        theta = acos( (a . b) / (|a| * |b|) ),
        with the cosine argument clamped to [-1, 1] before acos() to
        guard against floating-point rounding pushing it fractionally
        outside that domain (which would otherwise raise a domain error
        or, in some implementations, produce NaN).
    Engineering reasoning:
        Clamping is required, not optional: for near-parallel or
        near-anti-parallel vectors, floating-point error can produce a
        cosine of e.g. 1.0000000000000002, which `math.acos` rejects.
    """
    magnitude_a = a.magnitude()
    magnitude_b = b.magnitude()
    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0
    cosine = a.dot(b) / (magnitude_a * magnitude_b)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def heading(v: Vector3) -> float:
    """Compass-style heading of a vector in the ENU horizontal plane.

    Purpose:
        Reusable heading, delegating to `Vector3.heading`.
    Inputs:
        v: a `Vector3`; safe for the zero vector (see below).
    Outputs:
        float degrees in [0, 360).
    Units:
        degrees out.
    Mathematics:
        heading = atan2(vx, vy) mod 360
        (0 deg = +Y/North, 90 deg = +X/East, clockwise)
    Engineering reasoning:
        Delegates entirely to `Vector3.heading`, which is itself
        already NaN/exception-free for the zero vector: `atan2(0, 0)`
        is defined (as 0.0) rather than raising, unlike division-based
        formulas.
    """
    return v.heading()


def bearing(origin: Vector3, target: Vector3) -> float:
    """True (compass) bearing from `origin` to `target`, degrees, in [0, 360).

    Purpose:
        Reusable point-to-point compass bearing — new functionality;
        `Vector3` has no two-argument bearing method.
    Inputs:
        origin, target: `Vector3` positions, meters, ENU frame.
    Outputs:
        float degrees in [0, 360). Returns 0.0 if `origin == target`
        (bearing to a coincident point is undefined; 0.0 is the same
        safe default `Vector3.heading()` already gives the zero vector).
    Units:
        meters in, degrees out.
    Mathematics:
        bearing = heading(target - origin)
                = atan2((target-origin).x, (target-origin).y) mod 360
    Engineering reasoning:
        This is the *true* bearing (relative to North), independent of
        any observer heading. `GeometryEngine` builds the
        platform-relative `Bearing` field on top of this by subtracting
        Ownship's own heading.
    """
    return (target - origin).heading()
