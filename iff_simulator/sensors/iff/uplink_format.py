"""Logical uplink-format label attached to an interrogation message.

Purpose:
    Defines `UplinkFormat`, a *logical* label distinguishing kinds of
    uplink interrogation, and the simplified, fixed mapping from
    `IFFMode` to `UplinkFormat` used to populate it.

Inputs:
    An `IFFMode`.

Outputs:
    An `UplinkFormat` value.

Engineering explanation:
    These are logical values only — no real Mode S bit-field encoding
    (UF number, PC/AA/DI/AP fields, etc.) is implemented, per this
    phase's explicit "logical message objects only" constraint. Mode 5
    does not, in reality, have "Uplink Format" numbering the way Mode S
    does (that terminology is Mode-S-specific); the mapping below is a
    deliberate, documented simplification so every `InterrogationMessage`
    still has a deterministic `uplink_format` value regardless of mode,
    without inventing a second injectable policy the spec did not ask
    for. A future phase that models Mode S properly would replace this
    mapping, not extend it.
"""

from __future__ import annotations

from enum import Enum, unique

from .mode import IFFMode


@unique
class UplinkFormat(Enum):
    """Logical uplink-format identifiers only — no bit-field encoding.

    Purpose:
        Distinguish the "kind" of uplink interrogation at a logical
        level, matching this phase's "no real Mode S bit fields"
        constraint.
    """

    UF11 = "UF11"
    UF20 = "UF20"
    UF21 = "UF21"


#: Simplified, fixed IFFMode -> UplinkFormat mapping (see module docstring).
DEFAULT_UPLINK_FORMAT_BY_MODE: dict[IFFMode, UplinkFormat] = {
    IFFMode.MODE_S: UplinkFormat.UF11,
    IFFMode.MODE5_L1: UplinkFormat.UF20,
    IFFMode.MODE5_L2: UplinkFormat.UF21,
}
