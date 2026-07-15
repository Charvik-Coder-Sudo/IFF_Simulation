"""Phase 7 entry point: run the full receive/decode pipeline over recorded ground truth.

Purpose:
    Demonstrates and validates the Phase 7 architecture end to end:
    World -> TargetSelector -> InterrogationScheduler -> AirborneTransponder
    -> ReplyMatcher (propagation + timeout) -> ModeDecoder, producing one
    `DecodedIFFMeasurement` per interrogation, and writes them to
    `decoded_measurements.csv`.

Inputs:
    None (paths are derived relative to this file's location).

Outputs:
    iff_simulator/output/decoded_measurements.csv

Engineering explanation:
    Every aircraft in the recorded ground truth still defaults to
    `iff_capability="UNKNOWN"` and empty `mode_data` (no phase before
    this one assigns real IFF data), so this script — like
    `run_interrogation_scheduler.py` before it — rebuilds the loaded
    `Scenario`'s `Aircraft` list with demonstration IFF/Mode 5 data via
    `dataclasses.replace` (since `Aircraft` is frozen), touching no
    completed-phase file, purely so the receive pipeline has something
    non-empty to decode.

Validation:
    After the run, asserts that every interrogation produced exactly
    one decoded measurement (no duplicates, none dropped), that every
    measurement's status is VALID or NO_REPLY, and that
    measurement/sequence numbers strictly increase in the same order
    interrogations were issued (sequence ordering preserved).
"""

from __future__ import annotations

import csv
import dataclasses
from pathlib import Path

from iff_simulator.domain import Scenario
from iff_simulator.ground_truth import GroundTruthLoader
from iff_simulator.sensors.iff import (
    AirborneTransponder,
    InterrogationScheduler,
    MeasurementStatus,
    ModeDecoder,
    ReplyMatcher,
    TargetSelector,
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


def _measurement_to_csv_row(measurement) -> dict:
    return {
        "measurement_id": measurement.measurement_id,
        "time": measurement.time,
        "target_id": measurement.target_id,
        "ownship_id": measurement.ownship_id,
        "mode": measurement.mode.value,
        "range_m": measurement.range_m,
        "azimuth_deg": measurement.azimuth_deg,
        "elevation_deg": measurement.elevation_deg,
        "icao_address": measurement.icao_address,
        "authentication_result": measurement.authentication_result,
        "identity": measurement.identity,
        "mission": measurement.mission,
        "reply_status": measurement.reply_status.value,
        "processing_delay": measurement.processing_delay,
        "propagation_delay": measurement.propagation_delay,
        "arrival_time": measurement.arrival_time,
        "sequence_number": measurement.sequence_number,
    }


_CSV_COLUMNS = [
    "measurement_id", "time", "target_id", "ownship_id", "mode",
    "range_m", "azimuth_deg", "elevation_deg", "icao_address",
    "authentication_result", "identity", "mission", "reply_status",
    "processing_delay", "propagation_delay", "arrival_time", "sequence_number",
]


def _process_interrogation(interrogation, world, scenario, transponder, matcher, decoder):
    """Run one interrogation through Transponder -> ReplyMatcher -> ModeDecoder."""
    reply = transponder.receive(interrogation)
    ownship_position = world.ownship.position
    target_position = scenario.get_state(interrogation.target_id).position
    match_result = matcher.match(interrogation, reply, ownship_position, target_position)
    return decoder.decode(match_result)


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

    interrogations = []
    measurements = []

    first_interrogation = scheduler.tick()
    if first_interrogation is not None:
        interrogations.append(first_interrogation)
        measurements.append(
            _process_interrogation(first_interrogation, world, scenario, transponder, matcher, decoder)
        )

    while not clock.finished():
        world.step()
        interrogation = scheduler.tick()
        if interrogation is not None:
            interrogations.append(interrogation)
            measurements.append(
                _process_interrogation(interrogation, world, scenario, transponder, matcher, decoder)
            )

    output_path = OUTPUT_DIR / "decoded_measurements.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for measurement in measurements:
            writer.writerow(_measurement_to_csv_row(measurement))

    # --- Validation ---
    assert len(measurements) == len(interrogations), (
        f"Expected exactly one measurement per interrogation: "
        f"{len(interrogations)} interrogations, {len(measurements)} measurements"
    )

    for measurement in measurements:
        assert measurement.reply_status in (MeasurementStatus.VALID, MeasurementStatus.NO_REPLY)

    sequence_numbers = [m.sequence_number for m in measurements]
    assert sequence_numbers == sorted(sequence_numbers), "Sequence ordering was not preserved"
    assert len(set(sequence_numbers)) == len(sequence_numbers), "Duplicate sequence numbers detected"

    valid_count = sum(1 for m in measurements if m.reply_status == MeasurementStatus.VALID)
    no_reply_count = len(measurements) - valid_count

    print(f"Interrogations processed: {len(interrogations)}")
    print(f"Measurements produced:    {len(measurements)}")
    print(f"  VALID:    {valid_count}")
    print(f"  NO_REPLY: {no_reply_count}")
    print("Validation passed: exactly one decoded result per interrogation, "
          "all VALID/NO_REPLY, sequence ordering preserved, no duplicates.")
    print(f"Saved decoded measurements to {output_path}")


if __name__ == "__main__":
    main()
