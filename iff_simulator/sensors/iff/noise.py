"""Measurement noise on decoded range/azimuth/elevation only (Phase 9 Part 7).

Purpose:
    Implements `apply_measurement_noise`, which adds independent
    zero-mean Gaussian noise to a `DecodedIFFMeasurement`'s
    `range_m`/`azimuth_deg`/`elevation_deg` fields alone. This is
    measurement noise, not sensor fusion or filtering: each measurement
    is perturbed independently, with no state carried across ticks, no
    smoothing, and no estimation of any kind.

Inputs:
    A `DecodedIFFMeasurement`, three sigma values (meters/degrees/
    degrees), and the shared seeded `random.Random`.

Outputs:
    A new `DecodedIFFMeasurement` (via `dataclasses.replace` -- the
    original is frozen and never mutated) with noisy geometry fields;
    every other field (identity, authentication, timing, signal
    strength, ...) is copied verbatim.

Engineering explanation:
    Ground Truth is never read or touched here: `DecodedIFFMeasurement.
    range_m`/`azimuth_deg`/`elevation_deg` already come from the
    `InterrogationMessage` (itself `GeometryEngine`-derived, computed far
    upstream), not from `Scenario`/`Aircraft`/`AircraftState` directly --
    perturbing them cannot and does not affect Ground Truth in any way.
    Noise is only meaningful on a `VALID` measurement (a NO_REPLY/GARBLED/
    FRUITED measurement's geometry is either absent or not attributable
    to a real detection); callers should only invoke this on `VALID`
    measurements, but this function itself simply perturbs whatever
    numeric fields the measurement is holding.
"""

from __future__ import annotations

import dataclasses
import random

from .measurement import DecodedIFFMeasurement


def apply_measurement_noise(
    measurement: DecodedIFFMeasurement,
    sigma_range_m: float,
    sigma_azimuth_deg: float,
    sigma_elevation_deg: float,
    rng: random.Random,
) -> DecodedIFFMeasurement:
    """Return a copy of `measurement` with independent Gaussian noise
    added to range/azimuth/elevation.

    Inputs:
        measurement: the `DecodedIFFMeasurement` to perturb (unmodified;
            frozen dataclasses cannot be mutated in place).
        sigma_range_m, sigma_azimuth_deg, sigma_elevation_deg: standard
            deviations. A sigma of 0.0 adds no noise to that field
            (`rng.gauss(0.0, 0.0) == 0.0` always).
        rng: the shared seeded RNG (Part 12 determinism: never a private
            or global RNG).

    Outputs:
        A new `DecodedIFFMeasurement` with only `range_m`/`azimuth_deg`/
        `elevation_deg` changed.
    """
    return dataclasses.replace(
        measurement,
        range_m=measurement.range_m + rng.gauss(0.0, sigma_range_m),
        azimuth_deg=measurement.azimuth_deg + rng.gauss(0.0, sigma_azimuth_deg),
        elevation_deg=measurement.elevation_deg + rng.gauss(0.0, sigma_elevation_deg),
    )
