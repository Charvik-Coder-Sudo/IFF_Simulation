"""Tests for AuthenticationEngine."""

from __future__ import annotations

from iff_simulator.domain import Aircraft
from iff_simulator.sensors.iff import AuthenticationEngine


def _aircraft(**mode_data) -> Aircraft:
    return Aircraft(aircraft_id="T1", mode_data=mode_data)


def test_default_aircraft_is_not_authenticated():
    engine = AuthenticationEngine()
    assert engine.authenticate(_aircraft()) is False


def test_authenticated_requires_all_three_conditions():
    engine = AuthenticationEngine()
    fully_qualified = _aircraft(
        authentication_status="AUTHENTICATED", mode5_enabled=True, crypto_key_present=True
    )
    assert engine.authenticate(fully_qualified) is True


def test_wrong_authentication_status_fails():
    engine = AuthenticationEngine()
    aircraft = _aircraft(authentication_status="UNKNOWN", mode5_enabled=True, crypto_key_present=True)
    assert engine.authenticate(aircraft) is False


def test_mode5_disabled_fails_even_with_valid_status_and_key():
    engine = AuthenticationEngine()
    aircraft = _aircraft(authentication_status="AUTHENTICATED", mode5_enabled=False, crypto_key_present=True)
    assert engine.authenticate(aircraft) is False


def test_missing_crypto_key_fails_even_with_valid_status_and_mode5_enabled():
    engine = AuthenticationEngine()
    aircraft = _aircraft(authentication_status="AUTHENTICATED", mode5_enabled=True, crypto_key_present=False)
    assert engine.authenticate(aircraft) is False


def test_authenticate_is_deterministic():
    engine = AuthenticationEngine()
    aircraft = _aircraft(authentication_status="AUTHENTICATED", mode5_enabled=True, crypto_key_present=True)
    assert engine.authenticate(aircraft) is True
    assert engine.authenticate(aircraft) is True
