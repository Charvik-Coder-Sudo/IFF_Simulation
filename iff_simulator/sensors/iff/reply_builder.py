"""Dispatches an interrogation to the correct mode-specific reply generator.

Purpose:
    Implements `ReplyBuilder`, the single place that decides "Mode S or
    Mode 5?" and delegates to `ModeSReplyGenerator` or
    `Mode5ReplyGenerator` accordingly. `AirborneTransponder` calls this
    only after it has already decided a reply should be generated at
    all — `ReplyBuilder` itself never makes that decision.

Inputs:
    An `InterrogationMessage` and the target's `Aircraft` + `AircraftState`.

Outputs:
    A `ReplyMessage`.

Engineering explanation:
    Both generators are dependency-injected (defaulting to real
    instances), so a test can substitute a fake generator without
    touching `ReplyBuilder`'s dispatch logic.
"""

from __future__ import annotations

from ...domain import Aircraft, AircraftState
from .interrogation import InterrogationMessage
from .mode import IFFMode
from .mode5 import Mode5ReplyGenerator
from .mode_s import ModeSReplyGenerator
from .reply import ReplyMessage


class ReplyBuilder:
    """Dispatches to the correct mode-specific reply generator.

    Purpose:
        Decouple `AirborneTransponder` from knowing which generator
        class handles which `IFFMode`.

    Inputs:
        mode_s_generator: a `ModeSReplyGenerator` (dependency-injected;
            defaults to a real one).
        mode5_generator: a `Mode5ReplyGenerator` (dependency-injected;
            defaults to a real one).

    Outputs:
        `build(interrogation, aircraft, aircraft_state) -> ReplyMessage`.

    Engineering explanation:
        A thin Factory/dispatcher, not a decision-maker: by the time
        `build()` is called, `AirborneTransponder` has already decided
        a reply should exist — this class only decides *which*
        generator produces it.
    """

    def __init__(
        self,
        mode_s_generator: ModeSReplyGenerator | None = None,
        mode5_generator: Mode5ReplyGenerator | None = None,
    ) -> None:
        self.mode_s_generator = mode_s_generator or ModeSReplyGenerator()
        self.mode5_generator = mode5_generator or Mode5ReplyGenerator()

    def build(
        self,
        interrogation: InterrogationMessage,
        aircraft: Aircraft,
        aircraft_state: AircraftState,
    ) -> ReplyMessage:
        """Build the ReplyMessage for one interrogation.

        Inputs:
            interrogation: the `InterrogationMessage` to reply to.
            aircraft: the target's static `Aircraft` metadata.
            aircraft_state: the target's live `AircraftState`.
        Outputs:
            A `ReplyMessage` from the generator matching
            `interrogation.mode`.
        """
        if interrogation.mode == IFFMode.MODE_S:
            return self.mode_s_generator.generate(interrogation, aircraft, aircraft_state)
        return self.mode5_generator.generate(interrogation, aircraft, aircraft_state)
