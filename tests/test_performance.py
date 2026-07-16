"""Tests for Phase 9 Part 13: performance.

Covers: a procedurally generated (not .tdf-loaded) scenario with 300
aircraft, run through the full World -> Scheduler -> Transponder ->
ReceiverEffectsPipeline chain for a reduced-but-meaningful tick count
within a wall-clock budget. See receiver_pipeline.py's module docstring
for why the pipeline's own per-tick cost is O(1) regardless of aircraft
count -- this test demonstrates that at a scale large enough to catch
an accidental O(n) regression, without requiring the full 30,000-tick
run in every default test invocation.
"""

from __future__ import annotations

import time

import pytest

from iff_simulator.domain import Aircraft, AircraftState, Scenario, Vector3
from iff_simulator.sensors.iff import (
    AirborneTransponder,
    InterrogationScheduler,
    PD_MODEL_GAUSSIAN,
    ReceiverConfig,
    ReceiverEffectsPipeline,
    TargetSelector,
)
from iff_simulator.simulation import SimulationClock, World

ZERO = Vector3(0.0, 0.0, 0.0)


def _build_large_scenario(n_aircraft: int, n_samples: int) -> Scenario:
    aircraft = [Aircraft(aircraft_id="OWNSHIP")]
    history = {"OWNSHIP": [AircraftState(time=float(t), position=ZERO, velocity=ZERO) for t in range(n_samples)]}
    for i in range(n_aircraft):
        aircraft_id = f"TARGET_{i + 1}"
        aircraft.append(
            Aircraft(aircraft_id=aircraft_id, iff_capability="MODE_S_CAPABLE",
                      mode_data={"enabled_modes": ["MODE_S"]})
        )
        position = Vector3(100.0 + i * 5.0, 0.0, 0.0)
        history[aircraft_id] = [AircraftState(time=float(t), position=position, velocity=ZERO) for t in range(n_samples)]
    return Scenario(aircraft, history)


def _run(n_aircraft: int, n_ticks: int) -> float:
    scenario = _build_large_scenario(n_aircraft, n_ticks + 1)
    clock = SimulationClock(start_time=0.0, dt=1.0, end_time=float(n_ticks))
    world = World(scenario, clock, ownship_id="OWNSHIP", maximum_range=5000.0, beam_width=360.0,
                  beam_height=180.0, interrogation_rate=1.0)
    selector = TargetSelector(world)
    scheduler = InterrogationScheduler(world, selector)
    transponder = AirborneTransponder(scenario)
    config = ReceiverConfig(seed=1, pd_model=PD_MODEL_GAUSSIAN, pd_params={"r_max": 3000.0})
    pipeline = ReceiverEffectsPipeline(config=config, ownship_id="OWNSHIP", maximum_range_m=5000.0)

    start = time.perf_counter()
    interrogation = scheduler.tick()
    while True:
        reply = transponder.receive(interrogation) if interrogation is not None else None
        ownship_pos = world.ownship.position
        target_pos = (
            scenario.get_state(interrogation.target_id).position if interrogation is not None else ownship_pos
        )
        pipeline.process_tick(interrogation, reply, ownship_pos, target_pos, world.current_time())
        if clock.finished():
            break
        world.step()
        interrogation = scheduler.tick()
    return time.perf_counter() - start


def test_reduced_scale_runs_within_wall_clock_budget():
    """300 aircraft, 3,000 timesteps (1/10 of the Part 13 target tick
    count)."""
    elapsed = _run(n_aircraft=300, n_ticks=3000)
    assert elapsed < 30.0


@pytest.mark.skip(reason="Full Part 13 scale (300 aircraft x 30,000 timesteps) -- run manually to validate")
def test_full_scale_300_aircraft_30000_timesteps():
    elapsed = _run(n_aircraft=300, n_ticks=30_000)
    assert elapsed < 300.0
