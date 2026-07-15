"""Tests for Mode5ReplyGenerator, Mode5Level1Payload, Mode5Level2Payload."""

from __future__ import annotations

from iff_simulator.domain import Aircraft, AircraftState, Vector3
from iff_simulator.sensors.iff import (
    MISSION_TYPES,
    AuthenticationEngine,
    IFFMode,
    InterrogationMessage,
    Mode5Level1Payload,
    Mode5Level2Payload,
    Mode5ReplyGenerator,
    ReplyStatus,
    ReplyType,
    UplinkFormat,
    compute_mission_code,
)

ZERO = Vector3(0.0, 0.0, 0.0)

AUTHENTICATED_MODE_DATA = dict(
    authentication_status="AUTHENTICATED", mode5_enabled=True, crypto_key_present=True
)


def _interrogation(mode: IFFMode, sequence_number: int = 5) -> InterrogationMessage:
    uplink_format = UplinkFormat.UF20 if mode == IFFMode.MODE5_L1 else UplinkFormat.UF21
    return InterrogationMessage(
        time=20.0,
        sequence_number=sequence_number,
        ownship_id="OWNSHIP",
        target_id="T1",
        mode=mode,
        uplink_format=uplink_format,
        range_m=100.0,
        azimuth_deg=0.0,
        elevation_deg=0.0,
    )


def _aircraft(identity: str = "FRIEND", **mode_data) -> Aircraft:
    return Aircraft(aircraft_id="T1", identity=identity, iff_capability="MODE5_CAPABLE", mode_data=mode_data)


def _aircraft_state() -> AircraftState:
    return AircraftState(time=20.0, position=Vector3(0, 0, 1000), velocity=ZERO)


# ---------------------------------------------------------------------------
# Authenticated / unauthenticated
# ---------------------------------------------------------------------------


def test_authenticated_l1_reply_is_ok():
    generator = Mode5ReplyGenerator()
    aircraft = _aircraft(**AUTHENTICATED_MODE_DATA)
    reply = generator.generate(_interrogation(IFFMode.MODE5_L1), aircraft, _aircraft_state())
    assert reply.authenticated is True
    assert reply.reply_status == ReplyStatus.OK


def test_unauthenticated_l1_reply_still_generated_with_failed_auth_status():
    generator = Mode5ReplyGenerator()
    reply = generator.generate(_interrogation(IFFMode.MODE5_L1), _aircraft(), _aircraft_state())
    assert reply is not None
    assert reply.authenticated is False
    assert reply.reply_status == ReplyStatus.FAILED_AUTH


def test_unauthenticated_l2_reply_still_generated_with_failed_auth_status():
    generator = Mode5ReplyGenerator()
    reply = generator.generate(_interrogation(IFFMode.MODE5_L2), _aircraft(), _aircraft_state())
    assert reply.authenticated is False
    assert reply.reply_status == ReplyStatus.FAILED_AUTH


# ---------------------------------------------------------------------------
# Level 1
# ---------------------------------------------------------------------------


def test_l1_reply_has_level_1_and_correct_reply_type():
    generator = Mode5ReplyGenerator()
    reply = generator.generate(_interrogation(IFFMode.MODE5_L1), _aircraft(**AUTHENTICATED_MODE_DATA), _aircraft_state())
    assert reply.mode5_level == 1
    assert reply.reply_type == ReplyType.MODE5_L1_REPLY
    assert reply.mode_s_address is None


def test_l1_payload_fields():
    generator = Mode5ReplyGenerator()
    aircraft = _aircraft(
        identity="FRIEND",
        mission_code="ALPHA-7",
        platform_id="PLATFORM-99",
        **AUTHENTICATED_MODE_DATA,
    )
    reply = generator.generate(_interrogation(IFFMode.MODE5_L1), aircraft, _aircraft_state())
    assert isinstance(reply.payload, Mode5Level1Payload)
    assert reply.payload.authentication_result is True
    assert reply.payload.mission_code == "ALPHA-7"
    assert reply.payload.platform_id == "PLATFORM-99"
    assert reply.payload.friendly_status == "BLUE"  # authenticated -> BLUE, overriding "FRIEND"


def test_l1_friendly_status_falls_back_to_legacy_identity_when_unauthenticated():
    generator = Mode5ReplyGenerator()
    aircraft = _aircraft(identity="FOE")  # no AUTHENTICATED_MODE_DATA
    reply = generator.generate(_interrogation(IFFMode.MODE5_L1), aircraft, _aircraft_state())
    assert reply.authenticated is False
    assert reply.payload.friendly_status == "RED"


def test_l1_payload_defaults_when_mode_data_absent():
    generator = Mode5ReplyGenerator()
    aircraft = _aircraft()  # no mission_code/platform_id in mode_data
    reply = generator.generate(_interrogation(IFFMode.MODE5_L1), aircraft, _aircraft_state())
    assert reply.payload.mission_code == compute_mission_code("T1")
    assert reply.payload.mission_code in MISSION_TYPES
    assert reply.payload.platform_id == "T1"  # falls back to aircraft_id


# ---------------------------------------------------------------------------
# Level 2
# ---------------------------------------------------------------------------


def test_l2_reply_has_level_2_and_correct_reply_type():
    generator = Mode5ReplyGenerator()
    reply = generator.generate(_interrogation(IFFMode.MODE5_L2), _aircraft(**AUTHENTICATED_MODE_DATA), _aircraft_state())
    assert reply.mode5_level == 2
    assert reply.reply_type == ReplyType.MODE5_L2_REPLY


def test_l2_payload_fields():
    generator = Mode5ReplyGenerator()
    aircraft = _aircraft(
        platform_address="ADDR-123",
        mission="RECON",
        additional_status="LOW_FUEL",
        **AUTHENTICATED_MODE_DATA,
    )
    interrogation = _interrogation(IFFMode.MODE5_L2)
    reply = generator.generate(interrogation, aircraft, _aircraft_state())
    assert isinstance(reply.payload, Mode5Level2Payload)
    assert reply.payload.platform_address == "ADDR-123"
    assert reply.payload.mission == "RECON"
    assert reply.payload.time == interrogation.time
    assert reply.payload.additional_status == "LOW_FUEL"


def test_l2_payload_defaults_when_mode_data_absent():
    generator = Mode5ReplyGenerator()
    reply = generator.generate(_interrogation(IFFMode.MODE5_L2), _aircraft(), _aircraft_state())
    assert reply.payload.platform_address == "T1"
    assert reply.payload.mission == compute_mission_code("T1")
    assert reply.payload.mission in MISSION_TYPES
    assert reply.payload.additional_status == "NONE"


# ---------------------------------------------------------------------------
# Common
# ---------------------------------------------------------------------------


def test_processing_delay_is_75_microseconds():
    generator = Mode5ReplyGenerator()
    reply = generator.generate(_interrogation(IFFMode.MODE5_L1), _aircraft(**AUTHENTICATED_MODE_DATA), _aircraft_state())
    assert reply.processing_delay == 75.0


def test_preserves_sequence_ownship_target():
    generator = Mode5ReplyGenerator()
    interrogation = _interrogation(IFFMode.MODE5_L1, sequence_number=77)
    reply = generator.generate(interrogation, _aircraft(**AUTHENTICATED_MODE_DATA), _aircraft_state())
    assert reply.interrogation_sequence == 77
    assert reply.reply_id == 77
    assert reply.ownship_id == "OWNSHIP"
    assert reply.target_id == "T1"


def test_generate_is_deterministic():
    generator = Mode5ReplyGenerator(authentication_engine=AuthenticationEngine())
    interrogation = _interrogation(IFFMode.MODE5_L1)
    aircraft = _aircraft(**AUTHENTICATED_MODE_DATA)
    state = _aircraft_state()
    assert generator.generate(interrogation, aircraft, state) == generator.generate(interrogation, aircraft, state)
