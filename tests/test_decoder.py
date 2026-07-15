"""Tests for ModeDecoder."""

from __future__ import annotations

from iff_simulator.domain import Aircraft, AircraftState, Scenario, Vector3
from iff_simulator.sensors.iff import (
    AirborneTransponder,
    IFFMode,
    InterrogationMessage,
    MatchResult,
    MeasurementStatus,
    ModeDecoder,
    ReplyMatcher,
    UplinkFormat,
)

ZERO = Vector3(0.0, 0.0, 0.0)

AUTHENTICATED_MODE_DATA = dict(
    authentication_status="AUTHENTICATED", mode5_enabled=True, crypto_key_present=True
)


def _interrogation(mode: IFFMode, sequence_number: int = 1, time: float = 100.0) -> InterrogationMessage:
    uplink_format = {
        IFFMode.MODE_S: UplinkFormat.UF11,
        IFFMode.MODE5_L1: UplinkFormat.UF20,
        IFFMode.MODE5_L2: UplinkFormat.UF21,
    }[mode]
    return InterrogationMessage(
        time=time, sequence_number=sequence_number, ownship_id="OWNSHIP", target_id="T1",
        mode=mode, uplink_format=uplink_format, range_m=321.5, azimuth_deg=12.0, elevation_deg=3.0,
    )


def _scenario(enabled_modes, identity="FRIEND", altitude=5000.0, **extra_mode_data) -> Scenario:
    mode_data = {"enabled_modes": enabled_modes, **extra_mode_data}
    aircraft = [
        Aircraft(aircraft_id="OWNSHIP"),
        Aircraft(aircraft_id="T1", identity=identity, iff_capability="CAPABLE", mode_data=mode_data),
    ]
    history = {
        "OWNSHIP": [AircraftState(time=100.0, position=ZERO, velocity=ZERO)],
        "T1": [AircraftState(time=100.0, position=Vector3(100, 0, altitude), velocity=ZERO)],
    }
    return Scenario(aircraft, history)


def _decode(mode: IFFMode, scenario: Scenario, sequence_number: int = 1) -> MeasurementStatus:
    interrogation = _interrogation(mode, sequence_number=sequence_number)
    transponder = AirborneTransponder(scenario)
    reply = transponder.receive(interrogation)
    matcher = ReplyMatcher()
    match_result = matcher.match(
        interrogation, reply, scenario.get_state("OWNSHIP").position, scenario.get_state("T1").position
    )
    return ModeDecoder().decode(match_result)


# ---------------------------------------------------------------------------
# Mode S decode
# ---------------------------------------------------------------------------


def test_mode_s_decode_valid():
    scenario = _scenario(["MODE_S"])
    measurement = _decode(IFFMode.MODE_S, scenario)
    assert measurement.reply_status == MeasurementStatus.VALID
    assert measurement.icao_address is not None
    assert measurement.mission is None
    assert measurement.authentication_result is False


def test_mode_s_decode_geometry_copied_from_interrogation_not_recomputed():
    scenario = _scenario(["MODE_S"])
    measurement = _decode(IFFMode.MODE_S, scenario)
    assert measurement.range_m == 321.5
    assert measurement.azimuth_deg == 12.0
    assert measurement.elevation_deg == 3.0


def test_mode_s_decode_identity_classification():
    scenario = _scenario(["MODE_S"], identity="FOE")
    measurement = _decode(IFFMode.MODE_S, scenario)
    assert measurement.identity == "RED"


# ---------------------------------------------------------------------------
# Mode 5 L1 decode
# ---------------------------------------------------------------------------


def test_mode5_l1_decode_authenticated():
    scenario = _scenario(["MODE5_L1"], **AUTHENTICATED_MODE_DATA)
    measurement = _decode(IFFMode.MODE5_L1, scenario)
    assert measurement.reply_status == MeasurementStatus.VALID
    assert measurement.authentication_result is True
    assert measurement.identity == "BLUE"
    assert measurement.mission is not None
    assert measurement.icao_address is None


def test_mode5_l1_decode_unauthenticated_still_valid_measurement():
    scenario = _scenario(["MODE5_L1"])  # no auth mode_data
    measurement = _decode(IFFMode.MODE5_L1, scenario)
    assert measurement.reply_status == MeasurementStatus.VALID  # a reply WAS received
    assert measurement.authentication_result is False


# ---------------------------------------------------------------------------
# Mode 5 L2 decode
# ---------------------------------------------------------------------------


def test_mode5_l2_decode_authenticated():
    scenario = _scenario(["MODE5_L2"], **AUTHENTICATED_MODE_DATA)
    measurement = _decode(IFFMode.MODE5_L2, scenario)
    assert measurement.reply_status == MeasurementStatus.VALID
    assert measurement.authentication_result is True
    assert measurement.identity == "BLUE"
    assert measurement.mission is not None
    assert measurement.icao_address is None


def test_mode5_l2_decode_unauthenticated_identity_is_unknown():
    scenario = _scenario(["MODE5_L2"])
    measurement = _decode(IFFMode.MODE5_L2, scenario)
    assert measurement.authentication_result is False
    assert measurement.identity == "UNKNOWN"


# ---------------------------------------------------------------------------
# No Reply
# ---------------------------------------------------------------------------


def test_no_reply_when_transponder_never_responds():
    scenario = _scenario([])  # no modes enabled
    measurement = _decode(IFFMode.MODE_S, scenario)
    assert measurement.reply_status == MeasurementStatus.NO_REPLY
    assert measurement.icao_address is None
    assert measurement.mission is None
    assert measurement.identity == "UNKNOWN"
    assert measurement.authentication_result is False
    assert measurement.processing_delay is None
    assert measurement.propagation_delay is None
    assert measurement.arrival_time is None


def test_no_reply_preserves_geometry_and_identity_fields():
    interrogation = _interrogation(IFFMode.MODE_S, sequence_number=55)
    match_result = MatchResult(interrogation=interrogation, propagated_reply=None, timed_out=True)
    measurement = ModeDecoder().decode(match_result)
    assert measurement.sequence_number == 55
    assert measurement.measurement_id == 55
    assert measurement.target_id == "T1"
    assert measurement.ownship_id == "OWNSHIP"
    assert measurement.range_m == 321.5


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_decode_is_deterministic():
    scenario = _scenario(["MODE_S"])
    first = _decode(IFFMode.MODE_S, scenario)
    second = _decode(IFFMode.MODE_S, scenario)
    assert first == second


def test_sequence_integrity_measurement_id_matches_interrogation():
    scenario = _scenario(["MODE_S"])
    measurement = _decode(IFFMode.MODE_S, scenario, sequence_number=321)
    assert measurement.sequence_number == 321
    assert measurement.measurement_id == 321
