"""Tests for Phase 9 Part 11: ReceiverConfig.

Covers: every field defaults to its "effect disabled" value, every
field is independently overridable, and the dataclass is frozen.
"""

from __future__ import annotations

import dataclasses

import pytest

from iff_simulator.sensors.iff import ReceiverConfig


def test_defaults_disable_every_effect():
    config = ReceiverConfig()
    assert config.seed == 0
    assert config.pd_model == "always_detect"
    assert config.pd_params == {}
    assert config.pfa == 0.0
    assert config.sensitivity_threshold == 0.0
    assert config.capacity is None
    assert config.noise_sigma_range_m == 0.0
    assert config.noise_sigma_azimuth_deg == 0.0
    assert config.noise_sigma_elevation_deg == 0.0
    assert config.garble_window_s == 0.0
    assert config.fruiting_rate == 0.0
    assert config.jitter_processing_delay_us == 0.0
    assert config.jitter_propagation_delay_us == 0.0


def test_every_field_independently_overridable():
    config = ReceiverConfig(
        seed=7,
        pd_model="gaussian",
        pd_params={"r_max": 1000.0},
        pfa=0.1,
        sensitivity_threshold=0.2,
        capacity=5,
        noise_sigma_range_m=10.0,
        noise_sigma_azimuth_deg=1.0,
        noise_sigma_elevation_deg=2.0,
        garble_window_s=0.001,
        fruiting_rate=0.3,
        jitter_processing_delay_us=5.0,
        jitter_propagation_delay_us=3.0,
    )
    assert config.seed == 7
    assert config.pd_model == "gaussian"
    assert config.pd_params == {"r_max": 1000.0}
    assert config.pfa == 0.1
    assert config.sensitivity_threshold == 0.2
    assert config.capacity == 5
    assert config.noise_sigma_range_m == 10.0
    assert config.noise_sigma_azimuth_deg == 1.0
    assert config.noise_sigma_elevation_deg == 2.0
    assert config.garble_window_s == 0.001
    assert config.fruiting_rate == 0.3
    assert config.jitter_processing_delay_us == 5.0
    assert config.jitter_propagation_delay_us == 3.0


def test_config_is_frozen():
    config = ReceiverConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.seed = 99


def test_two_default_configs_are_equal():
    assert ReceiverConfig() == ReceiverConfig()
