"""Tests for iff_simulator.geometry.vector_math."""

from __future__ import annotations

import pytest

from iff_simulator.domain import Vector3
from iff_simulator.geometry import vector_math


def test_distance_matches_vector3_distance_to():
    a = Vector3(0, 0, 0)
    b = Vector3(3, 4, 0)
    assert vector_math.distance(a, b) == 5.0
    assert vector_math.distance(a, b) == a.distance_to(b)


def test_magnitude_matches_vector3_magnitude():
    v = Vector3(3, 4, 0)
    assert vector_math.magnitude(v) == v.magnitude() == 5.0


def test_normalize_matches_vector3_normalize():
    v = Vector3(0, 5, 0)
    assert vector_math.normalize(v) == v.normalize() == Vector3(0, 1, 0)


def test_normalize_raises_on_zero_vector():
    with pytest.raises(ValueError):
        vector_math.normalize(Vector3(0, 0, 0))


def test_safe_normalize_returns_zero_vector_instead_of_raising():
    result = vector_math.safe_normalize(Vector3(0, 0, 0))
    assert result == vector_math.ZERO_VECTOR
    assert result == Vector3(0.0, 0.0, 0.0)


def test_safe_normalize_matches_normalize_for_nonzero_vector():
    v = Vector3(0, 5, 0)
    assert vector_math.safe_normalize(v) == vector_math.normalize(v)


def test_dot_matches_vector3_dot():
    a = Vector3(1, 2, 3)
    b = Vector3(4, 5, 6)
    assert vector_math.dot(a, b) == a.dot(b) == 32.0


def test_cross_matches_vector3_cross():
    a = Vector3(1, 0, 0)
    b = Vector3(0, 1, 0)
    assert vector_math.cross(a, b) == a.cross(b) == Vector3(0, 0, 1)


def test_angle_between_parallel_vectors_is_zero():
    a = Vector3(1, 0, 0)
    b = Vector3(5, 0, 0)
    assert vector_math.angle_between(a, b) == pytest.approx(0.0, abs=1e-9)


def test_angle_between_perpendicular_vectors_is_90():
    a = Vector3(1, 0, 0)
    b = Vector3(0, 1, 0)
    assert vector_math.angle_between(a, b) == pytest.approx(90.0)


def test_angle_between_opposite_vectors_is_180():
    a = Vector3(1, 0, 0)
    b = Vector3(-1, 0, 0)
    assert vector_math.angle_between(a, b) == pytest.approx(180.0)


def test_angle_between_45_degrees():
    a = Vector3(1, 0, 0)
    b = Vector3(1, 1, 0)
    assert vector_math.angle_between(a, b) == pytest.approx(45.0)


def test_angle_between_zero_vector_is_safe_zero():
    a = Vector3(0, 0, 0)
    b = Vector3(1, 2, 3)
    assert vector_math.angle_between(a, b) == 0.0
    assert vector_math.angle_between(b, a) == 0.0
    assert vector_math.angle_between(a, a) == 0.0


def test_angle_between_near_parallel_does_not_raise_domain_error():
    # Constructed so the cosine is extremely close to 1.0 but may overshoot
    # due to floating-point rounding; must not raise a math domain error.
    a = Vector3(1.0, 1e-12, 0.0)
    b = Vector3(1.0, -1e-12, 0.0)
    result = vector_math.angle_between(a, b)
    assert result == pytest.approx(0.0, abs=1e-6)


def test_heading_matches_vector3_heading():
    v = Vector3(1, 0, 0)
    assert vector_math.heading(v) == v.heading() == 90.0


def test_heading_zero_vector_is_safe_zero():
    assert vector_math.heading(Vector3(0, 0, 0)) == 0.0


def test_bearing_north():
    origin = Vector3(0, 0, 0)
    target = Vector3(0, 100, 0)
    assert vector_math.bearing(origin, target) == pytest.approx(0.0)


def test_bearing_east():
    origin = Vector3(0, 0, 0)
    target = Vector3(100, 0, 0)
    assert vector_math.bearing(origin, target) == pytest.approx(90.0)


def test_bearing_south():
    origin = Vector3(0, 0, 0)
    target = Vector3(0, -100, 0)
    assert vector_math.bearing(origin, target) == pytest.approx(180.0)


def test_bearing_west():
    origin = Vector3(0, 0, 0)
    target = Vector3(-100, 0, 0)
    assert vector_math.bearing(origin, target) == pytest.approx(270.0)


def test_bearing_coincident_points_is_safe_zero():
    origin = Vector3(5, 5, 5)
    assert vector_math.bearing(origin, origin) == 0.0


def test_bearing_always_within_0_360():
    cases = [
        Vector3(1, 1, 0),
        Vector3(-1, 1, 0),
        Vector3(-1, -1, 0),
        Vector3(1, -1, 0),
    ]
    origin = Vector3(0, 0, 0)
    for target in cases:
        result = vector_math.bearing(origin, target)
        assert 0.0 <= result < 360.0
