"""Phase 10 entry point: run the realistic receiver-effects pipeline end to
end and analyze its output.

Purpose:
    Demonstrates and validates the Phase 10 architecture: drives the
    same World -> TargetSelector -> InterrogationScheduler ->
    AirborneTransponder -> ReceiverEffectsPipeline -> IFFTrackManager
    chain `run_receiver_pipeline.py` already exercises, but additionally
    captures every stage's output into a `PipelineRunRecord` and runs it
    through `AnalysisReportGenerator` to produce the six required CSVs,
    nine diagnostic plots, and one generated engineering-report summary.

Inputs:
    None (paths are derived relative to this file's location).

Outputs:
    iff_simulator/output/analysis/*.csv
    iff_simulator/output/analysis/engineering_report.md
    iff_simulator/output/analysis_plots/*.png

Engineering explanation:
    This script performs no simulation logic of its own beyond what
    `run_receiver_pipeline.py` already does -- it only additionally
    records each tick's `InterrogationMessage`/`ReplyMessage`/
    `ReceiverTickResult` into parallel lists so they can be bundled into
    a `PipelineRunRecord` afterward. No pipeline module is touched.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from iff_simulator.analysis import AnalysisReportGenerator, PipelineRunRecord
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
)
from iff_simulator.simulation import SimulationClock, World

PROJECT_ROOT = Path(__file__).resolve().parent
AIRCRAFTS_DIR = PROJECT_ROOT / "Aircrafts"
OUTPUT_DIR = PROJECT_ROOT / "iff_simulator" / "output" / "analysis"
PLOTS_DIR = PROJECT_ROOT / "iff_simulator" / "output" / "analysis_plots"

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
    garble_window_s=0.00001,
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
    track_manager = IFFTrackManager()

    interrogations = []
    replies = []
    tick_results = []

    def process(interrogation) -> None:
        if interrogation is None:
            return
        reply = transponder.receive(interrogation)
        ownship_position = world.ownship.position
        target_position = scenario.get_state(interrogation.target_id).position
        tick_result = pipeline.process_tick(
            interrogation, reply, ownship_position, target_position, world.current_time()
        )

        interrogations.append(interrogation)
        replies.append(reply)
        tick_results.append(tick_result)

        if tick_result.real_measurement is not None:
            track_manager.update(tick_result.real_measurement)
        for measurement in tick_result.false_alarm_measurements:
            track = track_manager.update(measurement)
            if track is not None and track.track_status == TrackStatus.LOST:
                pipeline.mark_phantom_resolved(measurement.target_id)

    process(scheduler.tick())
    while not clock.finished():
        world.step()
        process(scheduler.tick())

    record = PipelineRunRecord(
        scenario=scenario,
        interrogations=interrogations,
        replies=replies,
        tick_results=tick_results,
        active_tracks=track_manager.get_active_tracks(),
        completed_track_summaries=track_manager.get_completed_track_summaries(),
        receiver_statistics=pipeline.statistics.snapshot(),
    )

    generator = AnalysisReportGenerator(record)
    metrics = generator.compute_all()["performance_metrics"]

    csv_paths = generator.write_csv_outputs(OUTPUT_DIR)
    plot_paths = generator.write_plots(
        PLOTS_DIR, RECEIVER_CONFIG.pd_model, RECEIVER_CONFIG.pd_params, MAXIMUM_RANGE_M
    )
    report_path = generator.write_engineering_report(OUTPUT_DIR)

    print(f"Interrogations analyzed: {len(interrogations)}")
    print(f"  Detection Probability:        {metrics.detection_probability:.4f}")
    print(f"  False Alarm Rate:              {metrics.false_alarm_rate:.4f}")
    print(f"  Authentication Success Rate:   {metrics.authentication_success_rate:.4f}")
    print(f"  Reply Success Rate:            {metrics.reply_success_rate:.4f}")
    print(f"  Decoder Success Rate:          {metrics.decoder_success_rate:.4f}")
    print(f"  Track Confirmation Rate:       {metrics.track_confirmation_rate:.4f}")
    print(f"  Average Detection Range (m):   {metrics.average_detection_range_m:.2f}")
    print(f"  Average Signal Strength:       {metrics.average_signal_strength:.4f}")
    print(f"Saved {len(csv_paths)} CSV outputs to {OUTPUT_DIR}")
    print(f"Saved {len(plot_paths)} plots to {PLOTS_DIR}")
    print(f"Saved engineering report to {report_path}")


if __name__ == "__main__":
    main()
