"""Tests for Phase 9 Part 4: fruiting reply generation.

Covers: structural correctness, fresh phantom target_ids, synthetic
geometry bounds, and determinism. Classification as FRUITED (and never
being fed to IFFTrackManager) is covered by test_receiver_pipeline.py's
integration tests, since that behavior lives in ReceiverEffectsPipeline.
"""

from __future__ import annotations

import random

from iff_simulator.domain import Vector3
from iff_simulator.sensors.iff import FruitingGenerator, IFFMode, ReplyStatus
from iff_simulator.sensors.iff.mode_s import ModeSPayload


def test_fruited_reply_has_unknown_identity():
    generator = FruitingGenerator(random.Random(1), ownship_id="OWNSHIP", maximum_range_m=1000.0)
    fruited = generator.generate(time=1.0)
    reply = fruited.reply

    assert isinstance(reply.payload, ModeSPayload)
    assert reply.payload.identity == "UNKNOWN"
    assert reply.mode == IFFMode.MODE_S
    assert reply.reply_status == ReplyStatus.OK


def test_fruited_reply_target_id_is_phantom_and_fresh_each_call():
    generator = FruitingGenerator(random.Random(2), ownship_id="OWNSHIP", maximum_range_m=1000.0)
    first = generator.generate(time=1.0)
    second = generator.generate(time=2.0)
    assert first.reply.target_id.startswith("FRUIT-")
    assert second.reply.target_id.startswith("FRUIT-")
    assert first.reply.target_id != second.reply.target_id


def test_fruited_reply_synthetic_geometry_within_configured_envelope():
    generator = FruitingGenerator(random.Random(3), ownship_id="OWNSHIP", maximum_range_m=2000.0)
    for _ in range(50):
        fruited = generator.generate(time=1.0)
        assert 0.0 <= fruited.range_m <= 2000.0
        assert 0.0 <= fruited.azimuth_deg <= 360.0
        assert isinstance(fruited.target_position, Vector3)


def test_fruited_reply_generation_is_deterministic_given_same_seed():
    a = FruitingGenerator(random.Random(42), ownship_id="OWNSHIP", maximum_range_m=1000.0).generate(1.0)
    b = FruitingGenerator(random.Random(42), ownship_id="OWNSHIP", maximum_range_m=1000.0).generate(1.0)
    assert a.reply.reply_id == b.reply.reply_id
    assert a.range_m == b.range_m


def test_fruiting_and_false_reply_ids_do_not_collide_in_naming():
    """Different prefixes guarantee a fruited reply's target_id can never
    be mistaken for a false-alarm's, even under the same RNG stream."""
    generator = FruitingGenerator(random.Random(7), ownship_id="OWNSHIP", maximum_range_m=1000.0)
    fruited = generator.generate(time=1.0)
    assert not fruited.reply.target_id.startswith("FALSE-")
