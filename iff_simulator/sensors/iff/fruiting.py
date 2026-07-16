"""Fruiting reply generation (Phase 9 Part 4).

Purpose:
    Implements `FruitingGenerator`, which fabricates an asynchronous
    reply meant for a *different* interrogator's interrogation cycle,
    arriving at this receiver anyway -- exactly the real-world "fruit"
    phenomenon in a congested IFF/SSR environment. Structurally it looks
    like an ordinary reply (a real transponder somewhere really did
    reply to something), but it answers no interrogation this receiver
    ever transmitted, so the decoder classifies it `FRUITED` rather than
    `VALID` and it is never handed to `IFFTrackManager` (fruiting is a
    receiver/decoder classification concept only, per this phase's Part
    4 -- it is not meant to spawn tracks the way a false alarm does).

Inputs:
    A seeded `random.Random` (shared with the rest of
    `ReceiverEffectsPipeline`), the current simulation time, and the
    synthetic geometry envelope.

Outputs:
    A `FruitedReply` bundling the fabricated `ReplyMessage` with the
    synthetic position `ReplyPropagation.propagate` needs.

Engineering explanation:
    Mirrors `false_replies.py`'s construction shape closely (same
    "fabricate a plausible but synthetic reply, no real Ground Truth
    involved" approach) -- the two are kept as separate modules/classes
    because they mean different things downstream (Part 2's false alarms
    are track-worthy; Part 4's fruited replies are explicitly not), and
    conflating them into one generator would blur that distinction.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...domain import Vector3
from .false_replies import PROCESSING_DELAY_US, polar_to_vector3
from .mode import IFFMode
from .mode_s import ModeSPayload
from .reply import ReplyMessage, ReplyStatus, ReplyType

FRUIT_TARGET_ID_PREFIX = "FRUIT"
"""Prefix for every fabricated fruited-reply target_id, e.g. "FRUIT-1"."""


@dataclass(frozen=True, slots=True)
class FruitedReply:
    """A fabricated fruited reply, plus the synthetic geometry it
    "arrived from" for propagation purposes."""

    reply: ReplyMessage
    range_m: float
    azimuth_deg: float
    elevation_deg: float
    target_position: Vector3


class FruitingGenerator:
    """Fabricates asynchronous fruited replies (Part 4).

    Purpose:
        Produce a reply not tied to any interrogation this receiver
        transmitted, for the decoder to classify `FRUITED`.

    Inputs:
        rng: the shared, seeded `random.Random` instance (Part 12).
        ownship_id: the interrogating Ownship's aircraft_id.
        maximum_range_m: the synthetic geometry envelope.

    Outputs:
        `generate(time) -> FruitedReply`.
    """

    def __init__(self, rng, ownship_id: str, maximum_range_m: float = 5000.0) -> None:
        self.rng = rng
        self.ownship_id = ownship_id
        self.maximum_range_m = maximum_range_m
        self._next_id = 1

    def generate(self, time: float) -> FruitedReply:
        """Fabricate one fruited reply for the current tick."""
        target_id = f"{FRUIT_TARGET_ID_PREFIX}-{self._next_id}"
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
        return FruitedReply(
            reply=reply,
            range_m=range_m,
            azimuth_deg=azimuth_deg,
            elevation_deg=elevation_deg,
            target_position=target_position,
        )
