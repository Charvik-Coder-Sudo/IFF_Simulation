"""Tests for ReplyBuilder dispatch."""

from __future__ import annotations

from iff_simulator.domain import Aircraft, AircraftState, Vector3
from iff_simulator.sensors.iff import (
    IFFMode,
    InterrogationMessage,
    ModeSPayload,
    ReplyBuilder,
    UplinkFormat,
)

ZERO = Vector3(0.0, 0.0, 0.0)


def _interrogation(mode: IFFMode) -> InterrogationMessage:
    uplink_format = {
        IFFMode.MODE_S: UplinkFormat.UF11,
        IFFMode.MODE5_L1: UplinkFormat.UF20,
        IFFMode.MODE5_L2: UplinkFormat.UF21,
    }[mode]
    return InterrogationMessage(
        time=1.0, sequence_number=1, ownship_id="OWNSHIP", target_id="T1",
        mode=mode, uplink_format=uplink_format, range_m=100.0, azimuth_deg=0.0, elevation_deg=0.0,
    )


def _aircraft(**mode_data) -> Aircraft:
    return Aircraft(aircraft_id="T1", mode_data=mode_data)


def _aircraft_state() -> AircraftState:
    return AircraftState(time=1.0, position=ZERO, velocity=ZERO)


def test_mode_s_interrogation_dispatches_to_mode_s_generator():
    builder = ReplyBuilder()
    reply = builder.build(_interrogation(IFFMode.MODE_S), _aircraft(), _aircraft_state())
    assert isinstance(reply.payload, ModeSPayload)
    assert reply.mode == IFFMode.MODE_S


def test_mode5_l1_interrogation_dispatches_to_mode5_generator():
    builder = ReplyBuilder()
    reply = builder.build(_interrogation(IFFMode.MODE5_L1), _aircraft(), _aircraft_state())
    assert reply.mode5_level == 1


def test_mode5_l2_interrogation_dispatches_to_mode5_generator():
    builder = ReplyBuilder()
    reply = builder.build(_interrogation(IFFMode.MODE5_L2), _aircraft(), _aircraft_state())
    assert reply.mode5_level == 2
