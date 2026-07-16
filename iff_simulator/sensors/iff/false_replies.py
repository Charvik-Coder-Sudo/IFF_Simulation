"""False-alarm reply generation (Phase 9 Part 2).

Purpose:
    Implements `FalseReplyGenerator`, which fabricates a phantom
    `ReplyMessage` the receiver "hallucinates" — not answering any real
    interrogation, not tied to any real Ground Truth aircraft — with
    unknown ICAO, unknown authentication, a random sequence number, and
    unknown identity, exactly as this phase's Part 2 specifies. Because
    a false alarm is meant to look like an ordinary valid detection to
    the rest of the pipeline (so `IFFTrackManager` naturally starts a
    Tentative track for it), its payload shape is deliberately an
    ordinary `ModeSPayload` — just built from fabricated data instead of
    a real `Aircraft`.

Inputs:
    A seeded `random.Random` (shared with the rest of
    `ReceiverEffectsPipeline`, never a private/global RNG), the current
    simulation time, and a bound on the synthetic geometry envelope.

Outputs:
    A `(ReplyMessage, range_m, azimuth_deg, elevation_deg, target_position)`
    tuple: the fabricated reply plus a synthetic position for
    `ReplyPropagation.propagate` to compute an arrival time and signal
    strength from, exactly like a real reply.

Engineering explanation:
    The synthetic position is plain trigonometry over an RNG-drawn
    range/azimuth/elevation, not a `GeometryEngine` computation — there
    is no real Ground Truth aircraft here for `GeometryEngine` to
    describe; a false alarm has no real position, only a fabricated one
    the phantom "arrives from." No real Ground Truth (`Aircraft`,
    `AircraftState`, `Scenario`) is read, referenced, or mutated anywhere
    in this module.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from ...domain import Vector3
from .mode import IFFMode
from .mode_s import ModeSPayload
from .reply import ReplyMessage, ReplyStatus, ReplyType

FALSE_TARGET_ID_PREFIX = "FALSE"
"""Prefix for every fabricated false-alarm target_id, e.g. "FALSE-1",
"FALSE-2", ... — never collides with a real Scenario aircraft_id, which
this codebase's convention never prefixes this way."""

PROCESSING_DELAY_US = 50.0
"""Deterministic-shape processing delay for a false alarm, matching
`mode_s.PROCESSING_DELAY_US` -- a false alarm looks like an ordinary
Mode S reply to the rest of the pipeline."""


@dataclass(frozen=True, slots=True)
class FalseReply:
    """A fabricated false-alarm reply, plus the synthetic geometry it
    "arrived from" for propagation purposes.

    Purpose:
        Bundle everything `ReceiverEffectsPipeline` needs to propagate
        and decode a false alarm exactly like a real reply.
    """

    reply: ReplyMessage
    range_m: float
    azimuth_deg: float
    elevation_deg: float
    target_position: Vector3


class FalseReplyGenerator:
    """Fabricates phantom false-alarm replies (Part 2).

    Purpose:
        Produce a reply that looks, to the decoder and track manager,
        like an ordinary (if unidentified) valid detection -- unknown
        ICAO, unknown authentication, unknown identity, a random
        sequence number -- so `IFFTrackManager` naturally starts (and
        later loses) a Tentative track for it with zero changes to
        `IFFTrackManager` itself.

    Inputs:
        rng: the shared, seeded `random.Random` instance (Part 12:
            never a private or global RNG, so the whole pipeline stays
            reproducible from one seed).
        ownship_id: the interrogating Ownship's aircraft_id, copied onto
            the fabricated reply.
        maximum_range_m: the envelope the synthetic range is drawn from
            (matches the sensor's own configured maximum range so a
            false alarm looks plausible, never drawn from Ground Truth).

    Outputs:
        `generate(time) -> FalseReply`.
    """

    def __init__(self, rng: random.Random, ownship_id: str, maximum_range_m: float = 5000.0) -> None:
        self.rng = rng
        self.ownship_id = ownship_id
        self.maximum_range_m = maximum_range_m
        self._next_id = 1

    def generate(self, time: float) -> FalseReply:
        """Fabricate one false-alarm reply for the current tick.

        Inputs:
            time: current simulation time, copied onto the reply.

        Outputs:
            A `FalseReply` with a fresh `FALSE-<n>` target_id, a random
            (but RNG-seeded, hence reproducible) sequence number,
            `icao_address` that looks plausible but is not
            `compute_icao_address`-derived from any real aircraft_id,
            `identity="UNKNOWN"`, `capability="UNKNOWN"`, and
            `authenticated=False`.
        """
        target_id = f"{FALSE_TARGET_ID_PREFIX}-{self._next_id}"
        self._next_id += 1

        sequence_number = self.rng.randint(1, 0xFFFFFF)
        fake_icao = f"{self.rng.randint(0, 0xFFFFFF):06X}"

        range_m = self.rng.uniform(0.0, self.maximum_range_m)
        azimuth_deg = self.rng.uniform(0.0, 360.0)
        elevation_deg = self.rng.uniform(-10.0, 60.0)
        target_position = polar_to_vector3(range_m, azimuth_deg, elevation_deg)

        payload = ModeSPayload(
            icao_address=fake_icao,
            altitude_m=target_position.z,
            identity="UNKNOWN",
            capability="UNKNOWN",
            df_number=ReplyType.DF11.value,
        )

        reply = ReplyMessage(
            reply_id=sequence_number,
            time=time,
            interrogation_sequence=sequence_number,
            ownship_id=self.ownship_id,
            target_id=target_id,
            mode=IFFMode.MODE_S,
            reply_type=ReplyType.DF11,
            reply_status=ReplyStatus.OK,
            authenticated=False,
            mode_s_address=fake_icao,
            mode1=None,
            mode2=None,
            mode3A=None,
            modeC=None,
            mode5_level=None,
            payload=payload,
            processing_delay=PROCESSING_DELAY_US,
        )
        return FalseReply(
            reply=reply,
            range_m=range_m,
            azimuth_deg=azimuth_deg,
            elevation_deg=elevation_deg,
            target_position=target_position,
        )


def polar_to_vector3(range_m: float, azimuth_deg: float, elevation_deg: float) -> Vector3:
    """Plain trigonometry from (range, azimuth, elevation) to an ENU
    Vector3, for a fabricated position with no real Ground Truth
    behind it (never a `GeometryEngine` call -- there is nothing real
    for `GeometryEngine` to describe here)."""
    azimuth_rad = math.radians(azimuth_deg)
    elevation_rad = math.radians(elevation_deg)
    horizontal = range_m * math.cos(elevation_rad)
    x = horizontal * math.sin(azimuth_rad)
    y = horizontal * math.cos(azimuth_rad)
    z = range_m * math.sin(elevation_rad)
    return Vector3(x, y, z)
