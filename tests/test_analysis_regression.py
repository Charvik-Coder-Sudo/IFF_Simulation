"""Regression tests: importing/using `iff_simulator.analysis` must never
change the behavior of any completed pipeline module.

Rather than re-running the entire existing suite from inside a test
(the full suite itself is the authoritative regression check -- see
`docs/REGRESSION_SUMMARY.md`), this file makes direct, targeted
assertions that the specific things Phase 10 could plausibly have
disturbed are unchanged: the backward-compatibility invariant Phase 9
already established (`ReplyMatcher`+`ModeDecoder` produces identical
output to `ReceiverEffectsPipeline` with every effect off) still holds
with `iff_simulator.analysis` imported, and every core pipeline class
Phase 10 reads from is still the original, unshadowed class -- proving
the analysis package only ever *reads* pipeline output, never patches
or replaces pipeline behavior.
"""

from __future__ import annotations

import iff_simulator.analysis  # noqa: F401 -- import side effects are exactly what this file checks for
from iff_simulator.domain import Aircraft, AircraftState, Scenario, Vector3
from iff_simulator.sensors.iff import (
    AirborneTransponder,
    IFFTrackManager,
    InterrogationScheduler,
    ModeDecoder,
    Receiver,
    ReceiverConfig,
    ReceiverEffectsPipeline,
    ReplyMatcher,
    TargetSelector,
)
from iff_simulator.simulation import SimulationClock, World

ZERO = Vector3(0.0, 0.0, 0.0)


def _build_scenario():
    aircraft = [
        Aircraft(aircraft_id="OWNSHIP"),
        Aircraft(aircraft_id="T1", iff_capability="MODE_S_CAPABLE", mode_data={"enabled_modes": ["MODE_S"]}),
    ]
    history = {
        "OWNSHIP": [AircraftState(time=1.0, position=ZERO, velocity=ZERO)],
        "T1": [AircraftState(time=1.0, position=Vector3(100, 0, 0), velocity=Vector3(-5, 0, 0))],
    }
    return Scenario(aircraft, history)


def test_core_classes_are_not_shadowed_by_the_analysis_package():
    """`iff_simulator.analysis` only ever imports from `iff_simulator.sensors.iff`
    -- it must never monkeypatch or replace those classes."""
    assert ReceiverEffectsPipeline.__module__ == "iff_simulator.sensors.iff.receiver_pipeline"
    assert ModeDecoder.__module__ == "iff_simulator.sensors.iff.decoder"
    assert ReplyMatcher.__module__ == "iff_simulator.sensors.iff.matcher"
    assert Receiver.__module__ == "iff_simulator.sensors.iff.receiver"
    assert IFFTrackManager.__module__ == "iff_simulator.sensors.iff.track_manager"


def test_backward_compatible_pipeline_still_matches_with_analysis_imported():
    """Phase 9's own backward-compatibility invariant (an all-off
    `ReceiverConfig` reproduces `ReplyMatcher`+`ModeDecoder` exactly)
    must still hold with `iff_simulator.analysis` imported alongside."""
    scenario = _build_scenario()
    clock = SimulationClock(start_time=1.0, dt=1.0, end_time=3.0)
    world = World(
        scenario, clock, ownship_id="OWNSHIP", maximum_range=1000.0, beam_width=360.0,
        beam_height=180.0, interrogation_rate=1.0,
    )
    selector = TargetSelector(world)
    scheduler = InterrogationScheduler(world, selector)
    transponder = AirborneTransponder(scenario)

    interrogation = scheduler.tick()
    reply = transponder.receive(interrogation)
    ownship_position = world.ownship.position
    target_position = scenario.get_state(interrogation.target_id).position

    matcher = ReplyMatcher()
    decoder = ModeDecoder()
    match_result = matcher.match(interrogation, reply, ownship_position, target_position)
    legacy_measurement = decoder.decode(match_result)

    pipeline = ReceiverEffectsPipeline(config=ReceiverConfig(), ownship_id="OWNSHIP", maximum_range_m=1000.0)
    tick_result = pipeline.process_tick(interrogation, reply, ownship_position, target_position, world.current_time())

    assert tick_result.real_measurement == legacy_measurement


def test_analysis_package_never_registers_a_scenario_mutation():
    """`Scenario` exposes no runtime registration/mutation API that
    analysis code could have (mis)used; confirm the class shape is
    unchanged (still only the read-only query methods Phase 1 defined)."""
    scenario = _build_scenario()
    mutating_method_names = {"add_aircraft", "set_aircraft", "update_aircraft", "remove_aircraft"}
    assert not mutating_method_names & set(dir(scenario))
