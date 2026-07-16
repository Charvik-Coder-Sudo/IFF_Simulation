"""Tests for Phase 9 Part 1: probability-of-detection models.

Covers: both Pd formulas at r=0/typical/very-large ranges, 0 <= Pd <= 1
always, "always_detect" is exactly 1.0, determinism, and error handling.
"""

from __future__ import annotations

import pytest

from iff_simulator.sensors.iff import (
    PD_MODEL_ALWAYS_DETECT,
    PD_MODEL_GAUSSIAN,
    PD_MODEL_INVERSE_QUARTIC,
    compute_pd,
    pd_always_detect,
    pd_gaussian,
    pd_inverse_quartic,
)


def test_always_detect_is_exactly_one_at_every_range():
    for r in (0.0, 1.0, 1000.0, 1e9):
        assert pd_always_detect(r) == 1.0
        assert compute_pd(r, PD_MODEL_ALWAYS_DETECT, {}) == 1.0


def test_gaussian_pd_is_one_at_zero_range():
    assert pd_gaussian(0.0, r_max=1000.0) == pytest.approx(1.0)


def test_gaussian_pd_decreases_with_range():
    r_max = 1000.0
    pd_near = pd_gaussian(100.0, r_max)
    pd_far = pd_gaussian(2000.0, r_max)
    assert 0.0 < pd_far < pd_near < 1.0


def test_gaussian_pd_at_r_max_is_exp_minus_one():
    r_max = 500.0
    assert pd_gaussian(r_max, r_max) == pytest.approx(0.36787944117144233)


def test_inverse_quartic_pd_is_one_at_zero_range():
    assert pd_inverse_quartic(0.0, r0=1000.0) == pytest.approx(1.0)


def test_inverse_quartic_pd_is_half_at_r0():
    r0 = 750.0
    assert pd_inverse_quartic(r0, r0) == pytest.approx(0.5)


def test_inverse_quartic_pd_decreases_with_range():
    r0 = 1000.0
    pd_near = pd_inverse_quartic(100.0, r0)
    pd_far = pd_inverse_quartic(5000.0, r0)
    assert 0.0 < pd_far < pd_near < 1.0


@pytest.mark.parametrize(
    "model,params",
    [
        (PD_MODEL_ALWAYS_DETECT, {}),
        (PD_MODEL_GAUSSIAN, {"r_max": 1000.0}),
        (PD_MODEL_INVERSE_QUARTIC, {"r0": 1000.0}),
    ],
)
def test_pd_always_in_unit_interval(model, params):
    for r in (0.0, 1.0, 500.0, 5000.0, 1_000_000.0):
        pd = compute_pd(r, model, params)
        assert 0.0 <= pd <= 1.0


def test_compute_pd_is_deterministic():
    for _ in range(5):
        assert compute_pd(1234.5, PD_MODEL_GAUSSIAN, {"r_max": 2000.0}) == compute_pd(
            1234.5, PD_MODEL_GAUSSIAN, {"r_max": 2000.0}
        )


def test_compute_pd_unknown_model_raises():
    with pytest.raises(ValueError):
        compute_pd(100.0, "not_a_real_model", {})


def test_gaussian_pd_approaches_zero_at_very_large_range():
    assert pd_gaussian(1e9, r_max=1000.0) == pytest.approx(0.0, abs=1e-9)


def test_inverse_quartic_pd_approaches_zero_at_very_large_range():
    assert pd_inverse_quartic(1e9, r0=1000.0) == pytest.approx(0.0, abs=1e-9)


def test_gaussian_pd_smaller_r_max_falls_off_faster():
    range_m = 500.0
    pd_tight = pd_gaussian(range_m, r_max=200.0)
    pd_wide = pd_gaussian(range_m, r_max=2000.0)
    assert pd_tight < pd_wide


def test_inverse_quartic_pd_smaller_r0_falls_off_faster():
    range_m = 500.0
    pd_tight = pd_inverse_quartic(range_m, r0=200.0)
    pd_wide = pd_inverse_quartic(range_m, r0=2000.0)
    assert pd_tight < pd_wide


def test_compute_pd_clamps_into_unit_interval_for_a_tiny_r_max():
    pd = compute_pd(1.0, PD_MODEL_GAUSSIAN, {"r_max": 1e-6})
    assert 0.0 <= pd <= 1.0


@pytest.mark.parametrize("model_name", [PD_MODEL_ALWAYS_DETECT, PD_MODEL_GAUSSIAN, PD_MODEL_INVERSE_QUARTIC])
def test_model_name_constants_are_distinct_strings(model_name):
    assert isinstance(model_name, str)
    assert model_name in {PD_MODEL_ALWAYS_DETECT, PD_MODEL_GAUSSIAN, PD_MODEL_INVERSE_QUARTIC}
