"""Phase 8 / 8.5 entry point: run the full IFF pipeline through track management.

Purpose:
    Demonstrates and validates the Phase 8 + 8.5 architecture end to
    end: World -> TargetSelector -> InterrogationScheduler ->
    AirborneTransponder -> ReplyMatcher -> ModeDecoder -> IFFTrackManager
    -> ReportGenerator -> ReportWriter, producing `iff_track_file.csv`
    and (Phase 8.5 Part 6) `track_summary.csv`.

Inputs:
    None (paths are derived relative to this file's location).

Outputs:
    iff_simulator/output/iff_track_file.csv
    iff_simulator/output/track_summary.csv

Engineering explanation:
    Reuses the same demonstration IFF/Mode 5 capability overlay
    `run_interrogation_scheduler.py`/`run_receive_pipeline.py` already
    use (via `dataclasses.replace` on a freshly-loaded `Scenario`),
    touching no frozen-phase file. `track_manager.update(measurement)`
    is called with no manual `relative_velocity`/`signal_strength`
    overrides at all: as of Phase 8.5 Part 1, `DecodedIFFMeasurement`
    itself carries `relative_velocity`/`closing_velocity_mps`/
    `signal_strength` (threaded through from the single per-tick
    `GeometryEngine`/`ReplyPropagation` computation), so
    `IFFTrackManager` picks them up automatically — no second geometry
    computation appears anywhere in this script.
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
    ModeDecoder,
    ReplyMatcher,
    ReportGenerator,
    ReportWriter,
    TargetSelector,
    TrackStatus,
    write_track_summary_csv,
)
from iff_simulator.simulation import SimulationClock, World

PROJECT_ROOT = Path(__file__).resolve().parent
AIRCRAFTS_DIR = PROJECT_ROOT / "Aircrafts"
OUTPUT_DIR = PROJECT_ROOT / "iff_simulator" / "output"

OWNSHIP_ID = "TARGET_1"
DEMO_MODE_DATA = {
    "enabled_modes": ["MODE_S", "MODE5_L1", "MODE5_L2"],
    "authentication_status": "AUTHENTICATED",
    "mode5_enabled": True,
    "crypto_key_present": True,
}


def _with_demo_iff_capability(scenario: Scenario, ownship_id: str) -> Scenario:
    """Return a new Scenario with every non-Ownship aircraft marked IFF
    capable, for demonstration purposes only (see module docstring)."""
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
        maximum_range=2000.0, beam_width=360.0, beam_height=180.0, interrogation_rate=20.0,
    )
    target_selector = TargetSelector(world)
    scheduler = InterrogationScheduler(world, target_selector)
    transponder = AirborneTransponder(scenario)
    matcher = ReplyMatcher()
    decoder = ModeDecoder()
    track_manager = IFFTrackManager()

    tracks = []

    def _process(interrogation) -> None:
        if interrogation is None:
            return
        reply = transponder.receive(interrogation)
        ownship_position = world.ownship.position
        target_position = scenario.get_state(interrogation.target_id).position
        match_result = matcher.match(interrogation, reply, ownship_position, target_position)
        measurement = decoder.decode(match_result)
        track = track_manager.update(measurement)
        if track is not None:
            tracks.append(track)

    _process(scheduler.tick())
    while not clock.finished():
        world.step()
        _process(scheduler.tick())

    reports = ReportGenerator().generate_many(tracks)
    output_path = ReportWriter().write(reports, OUTPUT_DIR / "iff_track_file.csv")

    summaries = track_manager.get_completed_track_summaries()
    summary_path = write_track_summary_csv(summaries, OUTPUT_DIR / "track_summary.csv")

    active_tracks = track_manager.get_active_tracks()
    confirmed = sum(1 for t in active_tracks if t.track_status == TrackStatus.CONFIRMED)
    tentative = sum(1 for t in active_tracks if t.track_status == TrackStatus.TENTATIVE)

    # Demonstrate Phase 8.5 Part 1: relative geometry is present without
    # a second GeometryEngine call, and Part 5: bounded track history.
    if tracks:
        sample_track_id = tracks[-1].aircraft_id
        history = track_manager.get_track_history(sample_track_id)
        print(f"Sample track ({sample_track_id}) last update relative_velocity: "
              f"{tracks[-1].relative_velocity}, closing_velocity_mps: {tracks[-1].closing_velocity_mps}")
        print(f"Sample track ({sample_track_id}) history length: {len(history)} (max {20})")

    print(f"Track snapshots produced: {len(tracks)}")
    print(f"Active tracks at end:     {len(active_tracks)} (confirmed={confirmed}, tentative={tentative})")
    print(f"Completed (lost) tracks:  {len(summaries)}")
    print(f"Saved IFF track file to {output_path}")
    print(f"Saved track summary to {summary_path}")


if __name__ == "__main__":
    main()
