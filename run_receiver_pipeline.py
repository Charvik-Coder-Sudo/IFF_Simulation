"""Phase 9 entry point: run the realistic receiver-effects pipeline end to end.

Purpose:
    Demonstrates and validates the Phase 9 architecture: World ->
    TargetSelector -> InterrogationScheduler -> AirborneTransponder ->
    ReceiverEffectsPipeline (Propagation -> Pd -> Receiver -> Garbling ->
    Fruiting -> Reply Loss -> Decoder) -> IFFTrackManager, producing
    real, false-alarm, and fruited `DecodedIFFMeasurement`s, and writes
    `receiver_statistics.csv` plus the Part 10 diagnostic plots.

Inputs:
    None (paths are derived relative to this file's location).

Outputs:
    iff_simulator/output/receiver_statistics.csv
    iff_simulator/output/receiver_plots/*.png

Engineering explanation:
    Mirrors `run_receive_pipeline.py`'s structure and demo-IFF-capability
    convention exactly, swapping the plain `ReplyMatcher`/`ModeDecoder`
    pair for a `ReceiverEffectsPipeline` configured with a non-trivial
    `ReceiverConfig` (every effect enabled) so the run actually exercises
    Pd loss, false alarms, garbling, fruiting, saturation, noise, and
    jitter -- not just the all-off backward-compatible path (that path
    is covered by `test_receiver_pipeline.py`'s regression test instead).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from iff_simulator.domain import Scenario
from iff_simulator.ground_truth import GroundTruthLoader
from iff_simulator.sensors.iff import (
    AirborneTransponder,
    IFFTrackManager,
    InterrogationScheduler,
    PD_MODEL_GAUSSIAN,
    ReceiverConfig,
    ReceiverEffectsPipeline,
    TargetSelector,
    TrackStatus,
    write_receiver_statistics_csv,
)
from iff_simulator.simulation import SimulationClock, World
from iff_simulator.visualization import ReceiverEffectsPlotter

PROJECT_ROOT = Path(__file__).resolve().parent
AIRCRAFTS_DIR = PROJECT_ROOT / "Aircrafts"
OUTPUT_DIR = PROJECT_ROOT / "iff_simulator" / "output"
PLOTS_DIR = OUTPUT_DIR / "receiver_plots"

OWNSHIP_ID = "TARGET_1"
MAXIMUM_RANGE_M = 2000.0
DEMO_MODE_DATA = {
    "enabled_modes": ["MODE_S", "MODE5_L1", "MODE5_L2"],
    "authentication_status": "AUTHENTICATED",
    "mode5_enabled": True,
    "crypto_key_present": True,
}

RECEIVER_CONFIG = ReceiverConfig(
    seed=42,
    pd_model=PD_MODEL_GAUSSIAN,
    pd_params={"r_max": MAXIMUM_RANGE_M},
    pfa=0.02,
    sensitivity_threshold=0.05,
    capacity=10,
    noise_sigma_range_m=5.0,
    noise_sigma_azimuth_deg=0.5,
    noise_sigma_elevation_deg=0.5,
    garble_window_s=0.0002,
    fruiting_rate=0.05,
    jitter_processing_delay_us=5.0,
    jitter_propagation_delay_us=2.0,
)


def _with_demo_iff_capability(scenario: Scenario, ownship_id: str) -> Scenario:
    aircraft_list = [
        aircraft
        if aircraft.aircraft_id == ownship_id
        else dataclasses.replace(aircraft, iff_capability="MODE_S_CAPABLE", mode_data=DEMO_MODE_DATA)
        for aircraft in scenario.get_all_aircraft()
    ]
    state_history = {
        aircraft_id: scenario.get_state_history(aircraft_id)
        for aircraft_id in scenario.list_aircraft_ids()
    }
    return Scenario(aircraft_list, state_history)


def main() -> None:
    scenario = GroundTruthLoader(AIRCRAFTS_DIR).load()
    scenario = _with_demo_iff_capability(scenario, OWNSHIP_ID)

    ownship_history = scenario.get_state_history(OWNSHIP_ID)
    start_time = ownship_history[0].time
    end_time = ownship_history[-1].time
    dt = ownship_history[1].time - ownship_history[0].time

    clock = SimulationClock(start_time=start_time, dt=dt, end_time=end_time)
    world = World(
        scenario, clock, ownship_id=OWNSHIP_ID,
        maximum_range=MAXIMUM_RANGE_M, beam_width=360.0, beam_height=180.0, interrogation_rate=20.0,
    )
    target_selector = TargetSelector(world)
    scheduler = InterrogationScheduler(world, target_selector)
    transponder = AirborneTransponder(scenario)
    pipeline = ReceiverEffectsPipeline(
        config=RECEIVER_CONFIG, ownship_id=OWNSHIP_ID, maximum_range_m=MAXIMUM_RANGE_M
    )
    track_manager = IFFTrackManager(enable_logging=True)

    all_measurements = []

    def process(interrogation) -> None:
        reply = transponder.receive(interrogation) if interrogation is not None else None
        ownship_position = world.ownship.position
        target_position = (
            scenario.get_state(interrogation.target_id).position if interrogation is not None else ownship_position
        )
        tick_result = pipeline.process_tick(
            interrogation, reply, ownship_position, target_position, world.current_time()
        )

        if tick_result.real_measurement is not None:
            all_measurements.append(tick_result.real_measurement)
            track_manager.update(tick_result.real_measurement)

        for measurement in tick_result.false_alarm_measurements:
            all_measurements.append(measurement)
            track = track_manager.update(measurement)
            if track is not None and track.track_status == TrackStatus.LOST:
                pipeline.mark_phantom_resolved(measurement.target_id)

        all_measurements.extend(tick_result.fruited_measurements)

    first_interrogation = scheduler.tick()
    process(first_interrogation)
    while not clock.finished():
        world.step()
        process(scheduler.tick())

    stats = pipeline.statistics.snapshot()
    stats_path = write_receiver_statistics_csv(stats, OUTPUT_DIR / "receiver_statistics.csv")

    plotter = ReceiverEffectsPlotter(all_measurements, pipeline.statistics, track_manager.log)
    plot_paths = plotter.plot_all(
        PLOTS_DIR, RECEIVER_CONFIG.pd_model, RECEIVER_CONFIG.pd_params, MAXIMUM_RANGE_M
    )

    print(f"Measurements produced: {len(all_measurements)}")
    print(f"  Received: {stats.replies_received}")
    print(f"  Lost:     {stats.replies_lost}")
    print(f"  Garbled:  {stats.replies_garbled}")
    print(f"  Fruited:  {stats.replies_fruited}")
    print(f"  False:    {stats.false_replies}")
    print(f"  Avg Pd:              {stats.average_detection_probability:.4f}")
    print(f"  Avg signal strength: {stats.average_signal_strength:.4f}")
    print(f"  Avg delay (us):      {stats.average_delay_us:.2f}")
    print(f"  Receiver load:       {stats.receiver_load:.2f}")
    print(f"Saved receiver statistics to {stats_path}")
    print(f"Saved {len(plot_paths)} plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
