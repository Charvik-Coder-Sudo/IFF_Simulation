"""Tests for InterrogationMessage, IFFMode, UplinkFormat, and mode selection."""

from __future__ import annotations

import pytest

from iff_simulator.domain import Vector3
from iff_simulator.geometry import RelativeState
from iff_simulator.sensors.iff import (
    DEFAULT_UPLINK_FORMAT_BY_MODE,
    DefaultModeSelectionPolicy,
    IFFMode,
    InterrogationMessage,
    ModeSelectionPolicy,
    SelectedTarget,
    UplinkFormat,
)

ZERO = Vector3(0.0, 0.0, 0.0)


def _selected_target() -> SelectedTarget:
    relative_state = RelativeState(
        target_id="T1",
        time=12.5,
        relative_position=Vector3(100.0, 0.0, 0.0),
        relative_velocity=ZERO,
        range_m=100.0,
        azimuth_deg=15.0,
        elevation_deg=5.0,
        bearing_deg=15.0,
        closing_velocity_mps=3.0,
    )
    return SelectedTarget.from_relative_state(relative_state)


# ---------------------------------------------------------------------------
# IFFMode / UplinkFormat scope
# ---------------------------------------------------------------------------


def test_iff_mode_has_exactly_the_three_in_scope_modes():
    assert {mode.name for mode in IFFMode} == {"MODE_S", "MODE5_L1", "MODE5_L2"}


def test_iff_mode_excludes_legacy_modes():
    legacy_names = {"MODE_1", "MODE_2", "MODE_3A", "MODE_C", "MODE_4"}
    assert legacy_names.isdisjoint({mode.name for mode in IFFMode})


def test_uplink_format_has_exactly_three_logical_values():
    assert {uf.name for uf in UplinkFormat} == {"UF11", "UF20", "UF21"}


def test_default_uplink_format_mapping_covers_every_mode():
    assert set(DEFAULT_UPLINK_FORMAT_BY_MODE.keys()) == set(IFFMode)
    assert DEFAULT_UPLINK_FORMAT_BY_MODE[IFFMode.MODE_S] == UplinkFormat.UF11
    assert DEFAULT_UPLINK_FORMAT_BY_MODE[IFFMode.MODE5_L1] == UplinkFormat.UF20
    assert DEFAULT_UPLINK_FORMAT_BY_MODE[IFFMode.MODE5_L2] == UplinkFormat.UF21


# ---------------------------------------------------------------------------
# ModeSelectionPolicy
# ---------------------------------------------------------------------------


def test_mode_selection_policy_is_abstract():
    with pytest.raises(TypeError):
        ModeSelectionPolicy()  # type: ignore[abstract]


def test_default_mode_selection_policy_always_returns_mode_s():
    policy = DefaultModeSelectionPolicy()
    assert policy.select_mode(_selected_target()) == IFFMode.MODE_S


# ---------------------------------------------------------------------------
# InterrogationMessage
# ---------------------------------------------------------------------------


def test_from_selected_target_copies_geometry_and_time_verbatim():
    target = _selected_target()
    message = InterrogationMessage.from_selected_target(
        target,
        sequence_number=7,
        ownship_id="OWNSHIP",
        mode=IFFMode.MODE5_L1,
        uplink_format=UplinkFormat.UF20,
    )
    assert message.time == target.time
    assert message.range_m == target.range_m
    assert message.azimuth_deg == target.azimuth_deg
    assert message.elevation_deg == target.elevation_deg
    assert message.target_id == target.target_id
    assert message.sequence_number == 7
    assert message.ownship_id == "OWNSHIP"
    assert message.mode == IFFMode.MODE5_L1
    assert message.uplink_format == UplinkFormat.UF20


def test_interrogation_message_is_immutable():
    message = InterrogationMessage.from_selected_target(
        _selected_target(), sequence_number=1, ownship_id="OWNSHIP",
        mode=IFFMode.MODE_S, uplink_format=UplinkFormat.UF11,
    )
    with pytest.raises(Exception):
        message.sequence_number = 999


def test_to_csv_row_has_exact_spec_column_names():
    message = InterrogationMessage.from_selected_target(
        _selected_target(), sequence_number=3, ownship_id="OWNSHIP",
        mode=IFFMode.MODE5_L2, uplink_format=UplinkFormat.UF21,
    )
    row = message.to_csv_row()
    # Phase 8.5 Part 1 appends Closing_Velocity/Relative_Velocity after
    # the original 9 columns.
    assert list(row.keys()) == [
        "Time", "Sequence", "Ownship_ID", "Target_ID", "Mode", "UF", "Range", "Azimuth", "Elevation",
        "Closing_Velocity", "Relative_Velocity",
    ]
    assert row["Sequence"] == 3
    assert row["Mode"] == "MODE5_L2"
    assert row["UF"] == "UF21"
    assert row["Range"] == message.range_m
    assert row["Closing_Velocity"] == message.closing_velocity_mps
