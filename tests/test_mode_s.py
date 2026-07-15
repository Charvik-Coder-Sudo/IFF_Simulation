"""Tests for ModeSReplyGenerator and ModeSPayload."""

from __future__ import annotations

from iff_simulator.domain import Aircraft, AircraftState, Vector3
from iff_simulator.sensors.iff import (
    IFFMode,
    InterrogationMessage,
    ModeSPayload,
    ModeSReplyGenerator,
    ReplyStatus,
    ReplyType,
    UplinkFormat,
    compute_icao_address,
)

ZERO = Vector3(0.0, 0.0, 0.0)


def _interrogation(uplink_format: UplinkFormat = UplinkFormat.UF11, sequence_number: int = 5) -> InterrogationMessage:
    return InterrogationMessage(
        time=10.0,
        sequence_number=sequence_number,
        ownship_id="OWNSHIP",
        target_id="T1",
        mode=IFFMode.MODE_S,
        uplink_format=uplink_format,
        range_m=100.0,
        azimuth_deg=0.0,
        elevation_deg=0.0,
    )


def _aircraft(identity: str = "UNKNOWN", iff_capability: str = "MODE_S_CAPABLE") -> Aircraft:
    return Aircraft(aircraft_id="T1", identity=identity, iff_capability=iff_capability)


def _aircraft_state(altitude: float = 5000.0) -> AircraftState:
    return AircraftState(time=10.0, position=Vector3(0, 0, altitude), velocity=ZERO)


def test_generate_returns_ok_and_unauthenticated():
    generator = ModeSReplyGenerator()
    reply = generator.generate(_interrogation(), _aircraft(), _aircraft_state())
    assert reply.reply_status == ReplyStatus.OK
    assert reply.authenticated is False


def test_generate_preserves_sequence_ownship_target():
    generator = ModeSReplyGenerator()
    interrogation = _interrogation(sequence_number=42)
    reply = generator.generate(interrogation, _aircraft(), _aircraft_state())
    assert reply.interrogation_sequence == 42
    assert reply.reply_id == 42
    assert reply.ownship_id == "OWNSHIP"
    assert reply.target_id == "T1"
    assert reply.time == interrogation.time


def test_generate_maps_uplink_format_to_reply_type():
    generator = ModeSReplyGenerator()
    assert generator.generate(_interrogation(UplinkFormat.UF11), _aircraft(), _aircraft_state()).reply_type == ReplyType.DF11
    assert generator.generate(_interrogation(UplinkFormat.UF20), _aircraft(), _aircraft_state()).reply_type == ReplyType.DF20
    assert generator.generate(_interrogation(UplinkFormat.UF21), _aircraft(), _aircraft_state()).reply_type == ReplyType.DF21


def test_generate_sets_mode_s_address_and_no_mode5_fields():
    generator = ModeSReplyGenerator()
    reply = generator.generate(_interrogation(), _aircraft(), _aircraft_state())
    assert reply.mode_s_address == compute_icao_address("T1") == "A00001"
    assert reply.mode5_level is None
    assert reply.mode1 is None
    assert reply.mode2 is None
    assert reply.mode3A is None
    assert reply.modeC is None


def test_processing_delay_is_50_microseconds():
    generator = ModeSReplyGenerator()
    reply = generator.generate(_interrogation(), _aircraft(), _aircraft_state())
    assert reply.processing_delay == 50.0


def test_payload_altitude_comes_from_aircraft_state_position_z():
    generator = ModeSReplyGenerator()
    reply = generator.generate(_interrogation(), _aircraft(), _aircraft_state(altitude=12345.0))
    assert isinstance(reply.payload, ModeSPayload)
    assert reply.payload.altitude_m == 12345.0


def test_payload_identity_and_capability_come_from_aircraft():
    generator = ModeSReplyGenerator()
    aircraft = _aircraft(identity="FRIEND", iff_capability="LEVEL_2")
    reply = generator.generate(_interrogation(), aircraft, _aircraft_state())
    assert reply.payload.identity == "BLUE"  # legacy "FRIEND" -> BLUE via classify_friendly_status
    assert reply.payload.capability == "LEVEL_2"


def test_payload_identity_defaults_to_unknown_for_unrecognized_identity():
    generator = ModeSReplyGenerator()
    aircraft = _aircraft(identity="SOMETHING_ELSE")
    reply = generator.generate(_interrogation(), aircraft, _aircraft_state())
    assert reply.payload.identity == "UNKNOWN"


def test_payload_df_number_matches_reply_type():
    generator = ModeSReplyGenerator()
    reply = generator.generate(_interrogation(UplinkFormat.UF20), _aircraft(), _aircraft_state())
    assert reply.payload.df_number == "DF20" == reply.reply_type.value


def test_generate_is_deterministic():
    generator = ModeSReplyGenerator()
    interrogation = _interrogation()
    aircraft = _aircraft()
    state = _aircraft_state()
    reply_a = generator.generate(interrogation, aircraft, state)
    reply_b = generator.generate(interrogation, aircraft, state)
    assert reply_a == reply_b


# ---------------------------------------------------------------------------
# compute_icao_address
# ---------------------------------------------------------------------------


def test_compute_icao_address_matches_spec_examples_for_numbered_targets():
    assert compute_icao_address("TARGET_1") == "A00001"
    assert compute_icao_address("TARGET_2") == "A00002"
    assert compute_icao_address("TARGET_3") == "A00003"


def test_compute_icao_address_is_deterministic_and_stable():
    assert compute_icao_address("TARGET_1") == compute_icao_address("TARGET_1")


def test_compute_icao_address_is_unique_per_aircraft():
    assert compute_icao_address("TARGET_1") != compute_icao_address("TARGET_2")


def test_compute_icao_address_is_6_char_uppercase_hex():
    address = compute_icao_address("TARGET_1")
    assert len(address) == 6
    assert address == address.upper()
    int(address, 16)  # raises ValueError if not valid hex


def test_compute_icao_address_handles_ids_without_trailing_digits():
    address = compute_icao_address("OWNSHIP")
    assert len(address) == 6
    int(address, 16)
    assert address == compute_icao_address("OWNSHIP")  # stable across calls
