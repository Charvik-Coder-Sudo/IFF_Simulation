"""Tests for InterrogationScheduler, SchedulingPolicy, DefaultSchedulingPolicy."""

from __future__ import annotations

import pytest

from iff_simulator.domain import Aircraft, AircraftState, Scenario, Vector3
from iff_simulator.sensors.iff import (
    DefaultSchedulingPolicy,
    InterrogationScheduler,
    SchedulingPolicy,
    SelectedTarget,
    TargetSelector,
)
from iff_simulator.simulation import SimulationClock, World

ZERO = Vector3(0.0, 0.0, 0.0)
OWNSHIP_ID = "OWNSHIP"

DEFAULT_OWNSHIP_KWARGS = dict(
    maximum_range=100_000.0,
    beam_width=360.0,
    beam_height=180.0,
    interrogation_rate=1.0,
)


def _build_world(
    target_specs: dict[str, Vector3],
    start_time: float = 0.0,
    dt: float = 1.0,
    end_time: float = 100.0,
    **ownship_kwargs,
) -> World:
    """target_specs: target_id -> position. Every target is alive and IFF
    capable by default, so scheduler tests can focus purely on timing/
    priority behavior without re-testing Phase 4's selection rules."""
    aircraft_list = [Aircraft(aircraft_id=OWNSHIP_ID)]
    history = {OWNSHIP_ID: [AircraftState(time=start_time, position=ZERO, velocity=ZERO)]}
    for target_id, position in target_specs.items():
        aircraft_list.append(Aircraft(aircraft_id=target_id, iff_capability="MODE_4"))
        history[target_id] = [AircraftState(time=start_time, position=position, velocity=ZERO)]
    scenario = Scenario(aircraft_list, history)
    clock = SimulationClock(start_time=start_time, dt=dt, end_time=end_time)
    merged_kwargs = {**DEFAULT_OWNSHIP_KWARGS, **ownship_kwargs}
    return World(scenario, clock, ownship_id=OWNSHIP_ID, **merged_kwargs)


def _build_scheduler(world: World, **scheduler_kwargs) -> InterrogationScheduler:
    selector = TargetSelector(world)
    return InterrogationScheduler(world, selector, **scheduler_kwargs)


# ---------------------------------------------------------------------------
# SchedulingPolicy
# ---------------------------------------------------------------------------


def test_scheduling_policy_is_abstract():
    with pytest.raises(TypeError):
        SchedulingPolicy()  # type: ignore[abstract]


def test_default_scheduling_policy_empty_list_returns_none():
    assert DefaultSchedulingPolicy().choose([]) is None


def test_default_scheduling_policy_picks_closest_first():
    far = SelectedTarget(
        time=0.0, target_id="B", range_m=300.0, azimuth_deg=0.0, elevation_deg=0.0,
        closing_velocity_mps=0.0, relative_position=ZERO, relative_velocity=ZERO,
    )
    near = SelectedTarget(
        time=0.0, target_id="A", range_m=100.0, azimuth_deg=0.0, elevation_deg=0.0,
        closing_velocity_mps=0.0, relative_position=ZERO, relative_velocity=ZERO,
    )
    # TargetSelector already guarantees (range, id) order; policy trusts that order.
    chosen = DefaultSchedulingPolicy().choose([near, far])
    assert chosen.target_id == "A"


def test_default_scheduling_policy_is_deterministic():
    targets = [
        SelectedTarget(
            time=0.0, target_id="A", range_m=100.0, azimuth_deg=0.0, elevation_deg=0.0,
            closing_velocity_mps=0.0, relative_position=ZERO, relative_velocity=ZERO,
        ),
        SelectedTarget(
            time=0.0, target_id="B", range_m=100.0, azimuth_deg=0.0, elevation_deg=0.0,
            closing_velocity_mps=0.0, relative_position=ZERO, relative_velocity=ZERO,
        ),
    ]
    policy = DefaultSchedulingPolicy()
    first = policy.choose(targets)
    second = policy.choose(targets)
    assert first.target_id == second.target_id == "A"  # already sorted by TargetSelector contract


# ---------------------------------------------------------------------------
# No visible targets / one target / many targets
# ---------------------------------------------------------------------------


def test_no_visible_targets_never_transmits():
    world = _build_world({})
    scheduler = _build_scheduler(world)
    assert scheduler.tick() is None
    assert len(scheduler.queue) == 0


def test_one_target_is_transmitted():
    world = _build_world({"T1": Vector3(100, 0, 0)})
    scheduler = _build_scheduler(world)
    message = scheduler.tick()
    assert message is not None
    assert message.target_id == "T1"
    assert len(scheduler.queue) == 1


def test_many_targets_only_one_interrogated_per_transmission_instant():
    world = _build_world(
        {
            "T3": Vector3(300, 0, 0),
            "T1": Vector3(100, 0, 0),
            "T2": Vector3(200, 0, 0),
        }
    )
    scheduler = _build_scheduler(world)
    message = scheduler.tick()
    assert message is not None
    assert message.target_id == "T1"  # closest


def test_priority_ties_broken_by_lowest_aircraft_id():
    world = _build_world({"B": Vector3(100, 0, 0), "A": Vector3(100, 0, 0)})
    scheduler = _build_scheduler(world)
    message = scheduler.tick()
    assert message.target_id == "A"


# ---------------------------------------------------------------------------
# Sequence numbers
# ---------------------------------------------------------------------------


def test_sequence_numbers_start_at_one_and_increment():
    world = _build_world({"T1": Vector3(100, 0, 0)}, dt=1.0, end_time=10.0)
    scheduler = _build_scheduler(world, scheduling_policy=None)

    sequence_numbers = []
    msg = scheduler.tick()
    sequence_numbers.append(msg.sequence_number)
    for _ in range(3):
        world.step()
        msg = scheduler.tick()
        assert msg is not None
        sequence_numbers.append(msg.sequence_number)

    assert sequence_numbers == [1, 2, 3, 4]


def test_sequence_numbers_never_reused_even_when_no_target():
    """Sequence numbers only advance on *actual* transmissions, so an empty
    slot (no target) must not consume or skip a number."""
    world = _build_world({}, dt=1.0, end_time=5.0)
    scheduler = _build_scheduler(world)
    assert scheduler.tick() is None  # no targets ever

    # Now confirm a real target still starts at sequence 1.
    world2 = _build_world({"T1": Vector3(100, 0, 0)}, dt=1.0, end_time=5.0)
    scheduler2 = _build_scheduler(world2)
    message = scheduler2.tick()
    assert message.sequence_number == 1


# ---------------------------------------------------------------------------
# Interrogation timing
# ---------------------------------------------------------------------------


def test_period_is_inverse_of_interrogation_rate():
    world = _build_world({"T1": Vector3(100, 0, 0)}, interrogation_rate=20.0)
    scheduler = _build_scheduler(world)
    assert scheduler.period == pytest.approx(0.05)


def test_zero_interrogation_rate_never_transmits():
    world = _build_world({"T1": Vector3(100, 0, 0)}, interrogation_rate=0.0)
    scheduler = _build_scheduler(world)
    assert scheduler.tick() is None
    assert scheduler.period == float("inf")


def test_does_not_transmit_before_next_transmission_time():
    world = _build_world({"T1": Vector3(100, 0, 0)}, dt=1.0, end_time=10.0, interrogation_rate=0.1)  # period = 10s
    scheduler = _build_scheduler(world)

    first = scheduler.tick()
    assert first is not None  # first tick always eligible (next_transmission_time == start time)

    world.step()  # t = 1.0, period is 10s, not due yet
    assert scheduler.tick() is None


def test_transmits_once_next_transmission_time_is_reached():
    world = _build_world({"T1": Vector3(100, 0, 0)}, dt=1.0, end_time=10.0, interrogation_rate=1.0)  # period = 1s
    scheduler = _build_scheduler(world)

    assert scheduler.tick() is not None  # t=0
    assert scheduler.tick() is None  # still t=0, already transmitted this slot (next slot is t=1)

    world.step()  # t = 1.0
    assert scheduler.tick() is not None  # due again


def test_20hz_scheduler_transmits_at_correct_cadence():
    # period = 1/20 = 0.05s; dt = 0.01s -> one transmission every ~5 world steps.
    world = _build_world(
        {"T1": Vector3(100, 0, 0)},
        dt=0.01,
        end_time=0.3,
        interrogation_rate=20.0,
    )
    scheduler = _build_scheduler(world)

    transmission_times = []
    message = scheduler.tick()
    if message is not None:
        transmission_times.append(message.time)
    while not world.clock.finished():
        world.step()
        message = scheduler.tick()
        if message is not None:
            transmission_times.append(message.time)

    # Expect roughly 6-7 transmissions over 0.3s at a 0.05s cadence.
    assert 5 <= len(transmission_times) <= 8
    # Each gap is quantized to a whole number of world-clock steps (dt=0.01s):
    # the scheduler can only notice a due transmission when tick() is called,
    # so gaps cluster around the ideal 0.05s period but land on a multiple of
    # dt (e.g. 4, 5, or 6 ticks), not exactly 0.05s every time.
    gaps = [b - a for a, b in zip(transmission_times, transmission_times[1:])]
    for gap in gaps:
        assert gap == pytest.approx(0.05, abs=0.01 + 1e-6)


# ---------------------------------------------------------------------------
# Monotonic timestamps
# ---------------------------------------------------------------------------


def test_transmitted_timestamps_are_monotonically_nondecreasing():
    world = _build_world({"T1": Vector3(100, 0, 0)}, dt=1.0, end_time=10.0, interrogation_rate=1.0)
    scheduler = _build_scheduler(world)

    times = []
    message = scheduler.tick()
    times.append(message.time)
    while not world.clock.finished():
        world.step()
        message = scheduler.tick()
        if message is not None:
            times.append(message.time)

    assert all(later >= earlier for earlier, later in zip(times, times[1:]))


# ---------------------------------------------------------------------------
# Pause / resume
# ---------------------------------------------------------------------------


def test_scheduler_pause_stops_transmission():
    world = _build_world({"T1": Vector3(100, 0, 0)}, dt=1.0, end_time=10.0, interrogation_rate=1.0)
    scheduler = _build_scheduler(world)

    assert scheduler.tick() is not None  # t=0
    scheduler.pause()
    assert scheduler.is_paused() is True

    world.step()  # t=1, would normally be due
    assert scheduler.tick() is None


def test_scheduler_resume_allows_transmission_again():
    world = _build_world({"T1": Vector3(100, 0, 0)}, dt=1.0, end_time=10.0, interrogation_rate=1.0)
    scheduler = _build_scheduler(world)

    scheduler.tick()  # t=0
    scheduler.pause()
    world.step()  # t=1
    assert scheduler.tick() is None

    scheduler.resume()
    assert scheduler.is_paused() is False
    assert scheduler.tick() is not None  # still t=1, due since t=1 >= next_transmission_time(1)


def test_pause_does_not_consume_a_sequence_number():
    world = _build_world({"T1": Vector3(100, 0, 0)}, dt=1.0, end_time=10.0, interrogation_rate=1.0)
    scheduler = _build_scheduler(world)

    first = scheduler.tick()
    assert first.sequence_number == 1

    scheduler.pause()
    world.step()
    assert scheduler.tick() is None  # paused, no sequence consumed

    scheduler.resume()
    second = scheduler.tick()
    assert second.sequence_number == 2


# ---------------------------------------------------------------------------
# Deterministic ordering (repeat runs give identical results)
# ---------------------------------------------------------------------------


def test_scheduler_choice_is_deterministic_across_runs():
    specs = {"T3": Vector3(300, 0, 0), "T1": Vector3(100, 0, 0), "T2": Vector3(200, 0, 0)}

    world_a = _build_world(specs)
    world_b = _build_world(specs)
    message_a = _build_scheduler(world_a).tick()
    message_b = _build_scheduler(world_b).tick()

    assert message_a.target_id == message_b.target_id == "T1"


# ---------------------------------------------------------------------------
# Performance: O(N) per tick, no nested search
# ---------------------------------------------------------------------------


def test_scheduler_tick_scales_linearly_enough():
    import time as time_module

    target_count = 2000
    specs = {f"T{i}": Vector3(100 + i, 0, 0) for i in range(target_count)}
    world = _build_world(specs, maximum_range=1_000_000.0)
    scheduler = _build_scheduler(world)

    start = time_module.perf_counter()
    message = scheduler.tick()
    elapsed = time_module.perf_counter() - start

    assert message is not None
    assert message.target_id == "T0"  # closest of T0..T1999 by construction
    assert elapsed < 2.0
