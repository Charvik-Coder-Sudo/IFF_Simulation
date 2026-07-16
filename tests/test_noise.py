"""Tests for Phase 9 Part 7: measurement noise.

Covers: zero sigma -> zero change, statistical mean~=0 / std~=sigma over
many seeded samples, only range/azimuth/elevation change, and Ground
Truth is never touched (no Scenario/Aircraft/AircraftState involved).
"""

from __future__ import annotations

import random
import statistics

import pytest

from iff_simulator.sensors.iff import IFFMode, MeasurementStatus, apply_measurement_noise
from iff_simulator.sensors.iff.measurement import DecodedIFFMeasurement


def _measurement(range_m=1000.0, azimuth_deg=45.0, elevation_deg=10.0):
    return DecodedIFFMeasurement(
        measurement_id=1, time=1.0, target_id="T1", ownship_id="OWNSHIP", mode=IFFMode.MODE_S,
        range_m=range_m, azimuth_deg=azimuth_deg, elevation_deg=elevation_deg,
        icao_address="A00001", authentication_result=False, identity="BLUE", mission=None,
        reply_status=MeasurementStatus.VALID, processing_delay=50.0, propagation_delay=1.0,
        arrival_time=1.0, sequence_number=1, signal_strength=0.9,
    )


def test_zero_sigma_adds_no_noise():
    measurement = _measurement()
    rng = random.Random(1)
    noisy = apply_measurement_noise(measurement, 0.0, 0.0, 0.0, rng)
    assert noisy.range_m == measurement.range_m
    assert noisy.azimuth_deg == measurement.azimuth_deg
    assert noisy.elevation_deg == measurement.elevation_deg


def test_noise_only_changes_geometry_fields():
    measurement = _measurement()
    rng = random.Random(2)
    noisy = apply_measurement_noise(measurement, 5.0, 1.0, 1.0, rng)
    assert noisy.icao_address == measurement.icao_address
    assert noisy.identity == measurement.identity
    assert noisy.reply_status == measurement.reply_status
    assert noisy.signal_strength == measurement.signal_strength
    assert noisy.arrival_time == measurement.arrival_time
    assert noisy.target_id == measurement.target_id


def test_original_measurement_is_not_mutated():
    measurement = _measurement()
    rng = random.Random(3)
    apply_measurement_noise(measurement, 10.0, 2.0, 2.0, rng)
    assert measurement.range_m == 1000.0
    assert measurement.azimuth_deg == 45.0
    assert measurement.elevation_deg == 10.0


def test_noise_statistics_match_configured_sigma():
    measurement = _measurement(range_m=1000.0)
    rng = random.Random(1234)
    sigma = 10.0
    samples = [apply_measurement_noise(measurement, sigma, 0.0, 0.0, rng).range_m for _ in range(20_000)]
    deltas = [s - measurement.range_m for s in samples]
    assert statistics.mean(deltas) == pytest.approx(0.0, abs=0.5)
    assert statistics.pstdev(deltas) == pytest.approx(sigma, rel=0.05)


def test_noise_is_deterministic_given_same_seed():
    measurement = _measurement()
    a = apply_measurement_noise(measurement, 5.0, 1.0, 1.0, random.Random(55))
    b = apply_measurement_noise(measurement, 5.0, 1.0, 1.0, random.Random(55))
    assert a.range_m == b.range_m
    assert a.azimuth_deg == b.azimuth_deg
    assert a.elevation_deg == b.elevation_deg


def test_azimuth_noise_statistics_match_configured_sigma():
    measurement = _measurement(azimuth_deg=45.0)
    rng = random.Random(4321)
    sigma = 2.0
    samples = [apply_measurement_noise(measurement, 0.0, sigma, 0.0, rng).azimuth_deg for _ in range(20_000)]
    deltas = [s - measurement.azimuth_deg for s in samples]
    assert statistics.mean(deltas) == pytest.approx(0.0, abs=0.1)
    assert statistics.pstdev(deltas) == pytest.approx(sigma, rel=0.05)


def test_elevation_noise_statistics_match_configured_sigma():
    measurement = _measurement(elevation_deg=10.0)
    rng = random.Random(9876)
    sigma = 1.0
    samples = [apply_measurement_noise(measurement, 0.0, 0.0, sigma, rng).elevation_deg for _ in range(20_000)]
    deltas = [s - measurement.elevation_deg for s in samples]
    assert statistics.mean(deltas) == pytest.approx(0.0, abs=0.05)
    assert statistics.pstdev(deltas) == pytest.approx(sigma, rel=0.05)


def test_noise_applied_independently_per_axis():
    """A nonzero range sigma alone must not perturb azimuth/elevation."""
    measurement = _measurement()
    rng = random.Random(11)
    noisy = apply_measurement_noise(measurement, 50.0, 0.0, 0.0, rng)
    assert noisy.azimuth_deg == measurement.azimuth_deg
    assert noisy.elevation_deg == measurement.elevation_deg
