"""Exhaustive tests for AirborneTransponder."""

from __future__ import annotations

import pytest

from iff_simulator.domain import Aircraft, AircraftState, Scenario, Vector3
from iff_simulator.sensors.iff import (
    AirborneTransponder,
    IFFMode,
    InterrogationMessage,
    ReplyStatus,
    UplinkFormat,
    compute_icao_address,
)

ZERO = Vector3(0.0, 0.0, 0.0)

AUTHENTICATED_MODE_DATA = dict(
    authentication_status="AUTHENTICATED", mode5_enabled=True, crypto_key_present=True
)


def _build_scenario(
    alive: bool = True,
    iff_capability: str = "MODE_CAPABLE",
    enabled_modes: list[str] | None = None,
    identity: str = "FRIEND",
    **extra_mode_data,
) -> Scenario:
    mode_data = {"enabled_modes": enabled_modes or [], **extra_mode_data}
    aircraft = [
        Aircraft(aircraft_id="OWNSHIP"),
        Aircraft(aircraft_id="T1", identity=identity, iff_capability=iff_capability, mode_data=mode_data),
    ]
    history = {
        "OWNSHIP": [AircraftState(time=0.0, position=ZERO, velocity=ZERO)],
        "T1": [AircraftState(time=0.0, position=Vector3(0, 0, 5000), velocity=ZERO, alive=alive)],
    }
    return Scenario(aircraft, history)


def _interrogation(mode: IFFMode = IFFMode.MODE_S, sequence_number: int = 1) -> InterrogationMessage:
    uplink_format = {
        IFFMode.MODE_S: UplinkFormat.UF11,
        IFFMode.MODE5_L1: UplinkFormat.UF20,
        IFFMode.MODE5_L2: UplinkFormat.UF21,
    }[mode]
    return InterrogationMessage(
        time=0.0,
        sequence_number=sequence_number,
        ownship_id="OWNSHIP",
        target_id="T1",
        mode=mode,
        uplink_format=uplink_format,
        range_m=250.0,
        azimuth_deg=10.0,
        elevation_deg=2.0,
    )


# ---------------------------------------------------------------------------
# Alive / Dead
# ---------------------------------------------------------------------------


def test_alive_aircraft_can_reply():
    scenario = _build_scenario(alive=True, enabled_modes=["MODE_S"])
    transponder = AirborneTransponder(scenario)
    assert transponder.receive(_interrogation(IFFMode.MODE_S)) is not None


def test_dead_aircraft_never_replies():
    scenario = _build_scenario(alive=False, enabled_modes=["MODE_S"])
    transponder = AirborneTransponder(scenario)
    assert transponder.receive(_interrogation(IFFMode.MODE_S)) is None


# ---------------------------------------------------------------------------
# IFF capable / not capable
# ---------------------------------------------------------------------------


def test_iff_capable_aircraft_can_reply():
    scenario = _build_scenario(iff_capability="MODE_S_CAPABLE", enabled_modes=["MODE_S"])
    transponder = AirborneTransponder(scenario)
    assert transponder.receive(_interrogation(IFFMode.MODE_S)) is not None


def test_iff_incapable_aircraft_never_replies():
    scenario = _build_scenario(iff_capability="UNKNOWN", enabled_modes=["MODE_S"])
    transponder = AirborneTransponder(scenario)
    assert transponder.receive(_interrogation(IFFMode.MODE_S)) is None


# ---------------------------------------------------------------------------
# Mode enabled / disabled
# ---------------------------------------------------------------------------


def test_mode_enabled_can_reply():
    scenario = _build_scenario(enabled_modes=["MODE_S"])
    transponder = AirborneTransponder(scenario)
    assert transponder.receive(_interrogation(IFFMode.MODE_S)) is not None


def test_mode_disabled_never_replies():
    scenario = _build_scenario(enabled_modes=["MODE5_L1"])  # MODE_S not enabled
    transponder = AirborneTransponder(scenario)
    assert transponder.receive(_interrogation(IFFMode.MODE_S)) is None


def test_mode_enabled_is_per_mode_not_all_or_nothing():
    scenario = _build_scenario(enabled_modes=["MODE_S"])  # only MODE_S enabled
    transponder = AirborneTransponder(scenario)
    assert transponder.receive(_interrogation(IFFMode.MODE_S)) is not None
    assert transponder.receive(_interrogation(IFFMode.MODE5_L1, sequence_number=2)) is None


# ---------------------------------------------------------------------------
# Authenticated / unauthenticated (Mode 5, via full transponder path)
# ---------------------------------------------------------------------------


def test_mode5_authenticated_reply():
    scenario = _build_scenario(enabled_modes=["MODE5_L1"], **AUTHENTICATED_MODE_DATA)
    transponder = AirborneTransponder(scenario)
    reply = transponder.receive(_interrogation(IFFMode.MODE5_L1))
    assert reply is not None
    assert reply.authenticated is True
    assert reply.reply_status == ReplyStatus.OK


def test_mode5_unauthenticated_reply_still_generated():
    scenario = _build_scenario(enabled_modes=["MODE5_L1"])  # no auth mode_data
    transponder = AirborneTransponder(scenario)
    reply = transponder.receive(_interrogation(IFFMode.MODE5_L1))
    assert reply is not None
    assert reply.authenticated is False
    assert reply.reply_status == ReplyStatus.FAILED_AUTH


# ---------------------------------------------------------------------------
# Mode S / Mode5 L1 / Mode5 L2 end-to-end
# ---------------------------------------------------------------------------


def test_mode_s_end_to_end():
    scenario = _build_scenario(enabled_modes=["MODE_S"])
    transponder = AirborneTransponder(scenario)
    reply = transponder.receive(_interrogation(IFFMode.MODE_S))
    assert reply.mode == IFFMode.MODE_S
    assert reply.mode5_level is None
    assert reply.mode_s_address == compute_icao_address("T1")


def test_mode5_l1_end_to_end():
    scenario = _build_scenario(enabled_modes=["MODE5_L1"], **AUTHENTICATED_MODE_DATA)
    transponder = AirborneTransponder(scenario)
    reply = transponder.receive(_interrogation(IFFMode.MODE5_L1))
    assert reply.mode == IFFMode.MODE5_L1
    assert reply.mode5_level == 1


def test_mode5_l2_end_to_end():
    scenario = _build_scenario(enabled_modes=["MODE5_L2"], **AUTHENTICATED_MODE_DATA)
    transponder = AirborneTransponder(scenario)
    reply = transponder.receive(_interrogation(IFFMode.MODE5_L2))
    assert reply.mode == IFFMode.MODE5_L2
    assert reply.mode5_level == 2


# ---------------------------------------------------------------------------
# Processing delay
# ---------------------------------------------------------------------------


def test_mode_s_processing_delay():
    scenario = _build_scenario(enabled_modes=["MODE_S"])
    reply = AirborneTransponder(scenario).receive(_interrogation(IFFMode.MODE_S))
    assert reply.processing_delay == 50.0


def test_mode5_processing_delay():
    scenario = _build_scenario(enabled_modes=["MODE5_L1"], **AUTHENTICATED_MODE_DATA)
    reply = AirborneTransponder(scenario).receive(_interrogation(IFFMode.MODE5_L1))
    assert reply.processing_delay == 75.0


# ---------------------------------------------------------------------------
# Payload correctness (spot check through the full transponder path)
# ---------------------------------------------------------------------------


def test_payload_altitude_read_from_ground_truth_not_estimated():
    scenario = _build_scenario(enabled_modes=["MODE_S"])
    reply = AirborneTransponder(scenario).receive(_interrogation(IFFMode.MODE_S))
    # Ground truth position.z for T1 was set to 5000 in _build_scenario.
    assert reply.payload.altitude_m == 5000.0


# ---------------------------------------------------------------------------
# Sequence integrity
# ---------------------------------------------------------------------------


def test_sequence_target_ownship_preserved():
    scenario = _build_scenario(enabled_modes=["MODE_S"])
    transponder = AirborneTransponder(scenario)
    interrogation = _interrogation(IFFMode.MODE_S, sequence_number=999)
    reply = transponder.receive(interrogation)
    assert reply.interrogation_sequence == 999
    assert reply.reply_id == 999
    assert reply.target_id == interrogation.target_id == "T1"
    assert reply.ownship_id == interrogation.ownship_id == "OWNSHIP"


def test_sequence_integrity_across_multiple_interrogations():
    scenario = _build_scenario(enabled_modes=["MODE_S"])
    transponder = AirborneTransponder(scenario)
    replies = [transponder.receive(_interrogation(IFFMode.MODE_S, sequence_number=n)) for n in (1, 2, 3)]
    assert [r.interrogation_sequence for r in replies] == [1, 2, 3]
    assert [r.reply_id for r in replies] == [1, 2, 3]


# ---------------------------------------------------------------------------
# No Reply (aggregate)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(alive=False, enabled_modes=["MODE_S"]),
        dict(iff_capability="UNKNOWN", enabled_modes=["MODE_S"]),
        dict(enabled_modes=[]),
    ],
)
def test_no_reply_cases(kwargs):
    scenario = _build_scenario(**kwargs)
    transponder = AirborneTransponder(scenario)
    assert transponder.receive(_interrogation(IFFMode.MODE_S)) is None


# ---------------------------------------------------------------------------
# Determinism / reproducibility
# ---------------------------------------------------------------------------


def test_reply_is_reproducible_from_ground_truth_and_interrogation_alone():
    scenario = _build_scenario(enabled_modes=["MODE_S"])
    interrogation = _interrogation(IFFMode.MODE_S)

    reply_a = AirborneTransponder(scenario).receive(interrogation)
    reply_b = AirborneTransponder(scenario).receive(interrogation)  # fresh transponder instance

    assert reply_a == reply_b


def test_unknown_target_raises_key_error():
    scenario = _build_scenario(enabled_modes=["MODE_S"])
    transponder = AirborneTransponder(scenario)
    bad_interrogation = InterrogationMessage(
        time=0.0, sequence_number=1, ownship_id="OWNSHIP", target_id="NO_SUCH_TARGET",
        mode=IFFMode.MODE_S, uplink_format=UplinkFormat.UF11, range_m=1.0, azimuth_deg=0.0, elevation_deg=0.0,
    )
    with pytest.raises(KeyError):
        transponder.receive(bad_interrogation)
