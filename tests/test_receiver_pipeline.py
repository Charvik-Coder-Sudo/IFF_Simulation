"""Tests for Phase 9 ReceiverEffectsPipeline: integration, determinism,
backward compatibility, saturation, sensitivity, garbling, phantom
aging, and Ground Truth immutability.
"""

from __future__ import annotations

import dataclasses

import pytest

from iff_simulator.domain import Aircraft, AircraftState, Scenario, Vector3
from iff_simulator.sensors.iff import (
    AirborneTransponder,
    IFFMode,
    InterrogationMessage,
    InterrogationScheduler,
    MeasurementStatus,
    ModeDecoder,
    PD_MODEL_GAUSSIAN,
    ReceiverConfig,
    ReceiverEffectsPipeline,
    ReplyMatcher,
    ReplyMessage,
    ReplyStatus,
    ReplyType,
    TargetSelector,
    UplinkFormat,
    compute_signal_strength,
)
from iff_simulator.sensors.iff.mode_s import ModeSPayload
from iff_simulator.simulation import SimulationClock, World

ZERO = Vector3(0.0, 0.0, 0.0)


def _build_scenario_with_target(target_position=Vector3(100, 0, 0), target_velocity=Vector3(-5, 0, 0), n_samples=40):
    aircraft = [
        Aircraft(aircraft_id="OWNSHIP"),
        Aircraft(aircraft_id="T1", iff_capability="MODE_S_CAPABLE", mode_data={"enabled_modes": ["MODE_S"]}),
    ]
    history = {
        "OWNSHIP": [AircraftState(time=float(t), position=ZERO, velocity=ZERO) for t in range(1, n_samples)],
        "T1": [AircraftState(time=float(t), position=target_position, velocity=target_velocity) for t in range(1, n_samples)],
    }
    return Scenario(aircraft, history)


def _build_world(scenario):
    clock = SimulationClock(start_time=1.0, dt=1.0, end_time=20.0)
    world = World(scenario, clock, ownship_id="OWNSHIP", maximum_range=1000.0, beam_width=360.0,
                  beam_height=180.0, interrogation_rate=1.0)
    return clock, world


def _run_pipeline(config):
    scenario = _build_scenario_with_target()
    clock, world = _build_world(scenario)
    selector = TargetSelector(world)
    scheduler = InterrogationScheduler(world, selector)
    transponder = AirborneTransponder(scenario)
    pipeline = ReceiverEffectsPipeline(config=config, ownship_id="OWNSHIP", maximum_range_m=1000.0)

    tick_results = []
    interrogation = scheduler.tick()
    while True:
        reply = transponder.receive(interrogation) if interrogation is not None else None
        ownship_pos = world.ownship.position
        target_pos = scenario.get_state(interrogation.target_id).position if interrogation is not None else ownship_pos
        tick_results.append(pipeline.process_tick(interrogation, reply, ownship_pos, target_pos, world.current_time()))
        if clock.finished():
            break
        world.step()
        interrogation = scheduler.tick()
    return tick_results, pipeline, scenario


def _run_legacy_pipeline():
    scenario = _build_scenario_with_target()
    clock, world = _build_world(scenario)
    selector = TargetSelector(world)
    scheduler = InterrogationScheduler(world, selector)
    transponder = AirborneTransponder(scenario)
    matcher = ReplyMatcher()
    decoder = ModeDecoder()

    measurements = []
    interrogation = scheduler.tick()
    while True:
        if interrogation is not None:
            reply = transponder.receive(interrogation)
            ownship_pos = world.ownship.position
            target_pos = scenario.get_state(interrogation.target_id).position
            match_result = matcher.match(interrogation, reply, ownship_pos, target_pos)
            measurements.append(decoder.decode(match_result))
        else:
            measurements.append(None)
        if clock.finished():
            break
        world.step()
        interrogation = scheduler.tick()
    return measurements


def _interrogation(range_m, seq=1, time=1.0):
    return InterrogationMessage(
        time=time, sequence_number=seq, ownship_id="OWNSHIP", target_id="T1",
        mode=IFFMode.MODE_S, uplink_format=UplinkFormat.UF11,
        range_m=range_m, azimuth_deg=0.0, elevation_deg=0.0,
    )


def _reply_for(interrogation):
    payload = ModeSPayload(
        icao_address="A00001", altitude_m=0.0, identity="BLUE", capability="MODE_S_CAPABLE",
        df_number=ReplyType.DF11.value,
    )
    return ReplyMessage(
        reply_id=interrogation.sequence_number, time=interrogation.time,
        interrogation_sequence=interrogation.sequence_number, ownship_id=interrogation.ownship_id,
        target_id=interrogation.target_id, mode=interrogation.mode, reply_type=ReplyType.DF11,
        reply_status=ReplyStatus.OK, authenticated=False, mode_s_address="A00001",
        mode1=None, mode2=None, mode3A=None, modeC=None, mode5_level=None,
        payload=payload, processing_delay=50.0,
    )


# ---------------------------------------------------------------------------
# Backward compatibility / determinism (Part 12)
# ---------------------------------------------------------------------------


def test_default_config_matches_legacy_pipeline_field_for_field():
    tick_results, _pipeline, _scenario = _run_pipeline(ReceiverConfig())
    legacy_measurements = _run_legacy_pipeline()

    assert len(tick_results) == len(legacy_measurements)
    for tick_result, legacy in zip(tick_results, legacy_measurements):
        assert tick_result.real_measurement == legacy
        assert tick_result.false_alarm_measurements == []
        assert tick_result.fruited_measurements == []


def test_same_seed_same_config_produces_identical_output():
    config = ReceiverConfig(
        seed=123, pd_model=PD_MODEL_GAUSSIAN, pd_params={"r_max": 300.0}, pfa=0.4,
        fruiting_rate=0.4, garble_window_s=0.0005, jitter_processing_delay_us=10.0,
        jitter_propagation_delay_us=5.0, noise_sigma_range_m=5.0,
    )
    results_a, _, _ = _run_pipeline(config)
    results_b, _, _ = _run_pipeline(config)

    assert len(results_a) == len(results_b)
    for a, b in zip(results_a, results_b):
        assert a.real_measurement == b.real_measurement
        assert a.false_alarm_measurements == b.false_alarm_measurements
        assert a.fruited_measurements == b.fruited_measurements


def test_different_seed_changes_receiver_outcome():
    config_a = ReceiverConfig(seed=1, pfa=0.5, fruiting_rate=0.5)
    config_b = ReceiverConfig(seed=2, pfa=0.5, fruiting_rate=0.5)
    results_a, _, _ = _run_pipeline(config_a)
    results_b, _, _ = _run_pipeline(config_b)

    differs = any(
        a.real_measurement != b.real_measurement
        or a.false_alarm_measurements != b.false_alarm_measurements
        or a.fruited_measurements != b.fruited_measurements
        for a, b in zip(results_a, results_b)
    )
    assert differs


def test_ground_truth_never_mutated():
    scenario = _build_scenario_with_target()
    aircraft_before = scenario.get_aircraft("T1")
    clock, world = _build_world(scenario)
    selector = TargetSelector(world)
    scheduler = InterrogationScheduler(world, selector)
    transponder = AirborneTransponder(scenario)
    config = ReceiverConfig(seed=5, pfa=0.3, fruiting_rate=0.3, noise_sigma_range_m=5.0,
                             jitter_processing_delay_us=5.0)
    pipeline = ReceiverEffectsPipeline(config=config, ownship_id="OWNSHIP", maximum_range_m=1000.0)

    interrogation = scheduler.tick()
    while True:
        reply = transponder.receive(interrogation) if interrogation is not None else None
        ownship_pos = world.ownship.position
        target_pos = scenario.get_state(interrogation.target_id).position if interrogation is not None else ownship_pos
        pipeline.process_tick(interrogation, reply, ownship_pos, target_pos, world.current_time())
        if clock.finished():
            break
        world.step()
        interrogation = scheduler.tick()

    assert scenario.get_aircraft("T1") is aircraft_before
    assert scenario.get_aircraft("T1") == aircraft_before


# ---------------------------------------------------------------------------
# Sensitivity (Part 5)
# ---------------------------------------------------------------------------


def test_sensitivity_threshold_rejects_below_and_accepts_above():
    range_m = 2000.0
    signal_strength = compute_signal_strength(range_m)
    interrogation = _interrogation(range_m)
    reply = _reply_for(interrogation)
    target_position = Vector3(range_m, 0.0, 0.0)

    reject_config = ReceiverConfig(sensitivity_threshold=signal_strength + 0.01)
    reject_pipeline = ReceiverEffectsPipeline(config=reject_config, ownship_id="OWNSHIP", maximum_range_m=5000.0)
    reject_result = reject_pipeline.process_tick(interrogation, reply, ZERO, target_position, current_time=1.0)
    assert reject_result.real_measurement.reply_status == MeasurementStatus.NO_REPLY

    accept_config = ReceiverConfig(sensitivity_threshold=signal_strength - 0.01)
    accept_pipeline = ReceiverEffectsPipeline(config=accept_config, ownship_id="OWNSHIP", maximum_range_m=5000.0)
    accept_result = accept_pipeline.process_tick(interrogation, reply, ZERO, target_position, current_time=1.0)
    assert accept_result.real_measurement.reply_status == MeasurementStatus.VALID


# ---------------------------------------------------------------------------
# Saturation (Part 6)
# ---------------------------------------------------------------------------


def test_saturation_drops_replies_beyond_capacity():
    config = ReceiverConfig(seed=10, pfa=1.0, fruiting_rate=1.0, capacity=1)
    pipeline = ReceiverEffectsPipeline(config=config, ownship_id="OWNSHIP", maximum_range_m=1000.0)

    result = pipeline.process_tick(None, None, ZERO, ZERO, current_time=1.0)
    total_decoded = len(result.false_alarm_measurements) + len(result.fruited_measurements)

    assert total_decoded == 1
    assert pipeline.statistics.snapshot().replies_lost == 1


def test_no_capacity_configured_processes_everything():
    config = ReceiverConfig(seed=10, pfa=1.0, fruiting_rate=1.0, capacity=None)
    pipeline = ReceiverEffectsPipeline(config=config, ownship_id="OWNSHIP", maximum_range_m=1000.0)
    result = pipeline.process_tick(None, None, ZERO, ZERO, current_time=1.0)
    assert len(result.false_alarm_measurements) + len(result.fruited_measurements) == 2
    assert pipeline.statistics.snapshot().replies_lost == 0


# ---------------------------------------------------------------------------
# Garbling integration (Part 3)
# ---------------------------------------------------------------------------


def test_garbling_marks_real_reply_garbled_when_false_alarm_collides():
    range_m = 500.0
    interrogation = _interrogation(range_m, seq=1, time=1.0)
    reply = _reply_for(interrogation)
    target_position = Vector3(range_m, 0.0, 0.0)

    config = ReceiverConfig(seed=13, pfa=1.0, garble_window_s=10.0)
    pipeline = ReceiverEffectsPipeline(config=config, ownship_id="OWNSHIP", maximum_range_m=1000.0)

    result = pipeline.process_tick(interrogation, reply, ZERO, target_position, current_time=1.0)
    assert result.real_measurement.reply_status == MeasurementStatus.GARBLED
    assert pipeline.statistics.snapshot().replies_garbled == 2
    assert result.false_alarm_measurements == []  # the false alarm was garbled, not decoded VALID


# ---------------------------------------------------------------------------
# Fruiting integration (Part 4)
# ---------------------------------------------------------------------------


def test_fruited_measurements_are_kept_separate_from_false_alarms():
    config = ReceiverConfig(seed=12, fruiting_rate=1.0)
    pipeline = ReceiverEffectsPipeline(config=config, ownship_id="OWNSHIP", maximum_range_m=1000.0)
    result = pipeline.process_tick(None, None, ZERO, ZERO, current_time=1.0)

    assert len(result.fruited_measurements) == 1
    assert result.fruited_measurements[0].reply_status == MeasurementStatus.FRUITED
    assert result.false_alarm_measurements == []


# ---------------------------------------------------------------------------
# False-alarm phantom aging (Part 2)
# ---------------------------------------------------------------------------


def test_phantom_aging_continues_until_resolved():
    config = ReceiverConfig(seed=11, pfa=1.0)
    pipeline = ReceiverEffectsPipeline(config=config, ownship_id="OWNSHIP", maximum_range_m=1000.0)

    result1 = pipeline.process_tick(None, None, ZERO, ZERO, current_time=1.0)
    assert len(result1.false_alarm_measurements) == 1
    phantom_id = result1.false_alarm_measurements[0].target_id
    assert phantom_id in pipeline.active_phantom_ids()

    pipeline.config = dataclasses.replace(pipeline.config, pfa=0.0)
    result2 = pipeline.process_tick(None, None, ZERO, ZERO, current_time=2.0)
    aging_entries = [m for m in result2.false_alarm_measurements if m.target_id == phantom_id]
    assert len(aging_entries) == 1
    assert aging_entries[0].reply_status == MeasurementStatus.NO_REPLY

    pipeline.mark_phantom_resolved(phantom_id)
    assert phantom_id not in pipeline.active_phantom_ids()

    result3 = pipeline.process_tick(None, None, ZERO, ZERO, current_time=3.0)
    assert all(m.target_id != phantom_id for m in result3.false_alarm_measurements)


# ---------------------------------------------------------------------------
# Probability of Detection integration (Part 1)
# ---------------------------------------------------------------------------


def test_pd_zero_at_effectively_infinite_range_always_loses_reply():
    interrogation = _interrogation(range_m=1_000_000.0)
    reply = _reply_for(interrogation)
    target_position = Vector3(1_000_000.0, 0.0, 0.0)

    config = ReceiverConfig(pd_model=PD_MODEL_GAUSSIAN, pd_params={"r_max": 100.0})
    pipeline = ReceiverEffectsPipeline(config=config, ownship_id="OWNSHIP", maximum_range_m=2_000_000.0)
    result = pipeline.process_tick(interrogation, reply, ZERO, target_position, current_time=1.0)

    assert result.real_measurement.reply_status == MeasurementStatus.NO_REPLY
    assert pipeline.statistics.snapshot().replies_lost == 1


# ---------------------------------------------------------------------------
# Miscellaneous edge cases
# ---------------------------------------------------------------------------


def test_no_interrogation_no_reply_produces_no_real_measurement():
    config = ReceiverConfig()
    pipeline = ReceiverEffectsPipeline(config=config, ownship_id="OWNSHIP", maximum_range_m=1000.0)
    result = pipeline.process_tick(None, None, ZERO, ZERO, current_time=1.0)
    assert result.real_measurement is None
    assert result.false_alarm_measurements == []
    assert result.fruited_measurements == []


def test_interrogation_with_transponder_no_reply_resolves_to_no_reply():
    interrogation = _interrogation(range_m=500.0)
    config = ReceiverConfig()
    pipeline = ReceiverEffectsPipeline(config=config, ownship_id="OWNSHIP", maximum_range_m=1000.0)
    result = pipeline.process_tick(interrogation, None, ZERO, Vector3(500.0, 0.0, 0.0), current_time=1.0)
    assert result.real_measurement.reply_status == MeasurementStatus.NO_REPLY
    assert pipeline.statistics.snapshot().replies_lost == 1
    assert pipeline.statistics.snapshot().replies_received == 0


def test_capacity_zero_drops_every_reply_this_tick():
    config = ReceiverConfig(seed=15, pfa=1.0, capacity=0)
    pipeline = ReceiverEffectsPipeline(config=config, ownship_id="OWNSHIP", maximum_range_m=1000.0)
    result = pipeline.process_tick(None, None, ZERO, ZERO, current_time=1.0)
    assert result.false_alarm_measurements == []
    assert pipeline.statistics.snapshot().replies_lost == 1


def test_statistics_received_plus_lost_plus_garbled_covers_every_real_tick():
    config = ReceiverConfig(seed=20, pd_model=PD_MODEL_GAUSSIAN, pd_params={"r_max": 200.0})
    tick_results, pipeline, _scenario = _run_pipeline(config)
    real_ticks = [r for r in tick_results if r.real_measurement is not None]
    stats = pipeline.statistics.snapshot()
    assert stats.replies_received + stats.replies_lost == len(real_ticks)
