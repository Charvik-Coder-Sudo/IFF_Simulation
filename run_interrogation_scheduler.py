"""Phase 5 entry point: run the InterrogationScheduler over recorded ground truth.

Purpose:
    Demonstrates the Phase 5 architecture end to end: World ->
    TargetSelector -> InterrogationScheduler, stepping through the
    entire recorded ground-truth time range and writing every
    transmitted interrogation to `interrogations.csv`.

Inputs:
    None (paths are derived relative to this file's location).

Outputs:
    iff_simulator/output/interrogations.csv

Engineering explanation:
    No aircraft in the recorded ground truth has ever had its IFF
    capability assigned (`Aircraft.iff_capability` defaults to
    "UNKNOWN" — no phase before this one implements IFF identity
    logic), so `TargetSelector` would correctly select nothing and this
    demo would produce an empty CSV. To produce a meaningful,
    non-empty demonstration without touching any completed phase, this
    script — and only this script — rebuilds the loaded `Scenario`'s
    `Aircraft` list with a demonstration IFF capability
    (`dataclasses.replace`, since `Aircraft` is frozen) assigned to
    every non-Ownship aircraft. This does not modify `GroundTruthLoader`,
    `Aircraft`, `Scenario`, or any other completed-phase file.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from iff_simulator.domain import Scenario
from iff_simulator.ground_truth import GroundTruthLoader
from iff_simulator.sensors.iff import InterrogationScheduler, TargetSelector, write_interrogations_csv
from iff_simulator.simulation import SimulationClock, World

PROJECT_ROOT = Path(__file__).resolve().parent
AIRCRAFTS_DIR = PROJECT_ROOT / "Aircrafts"
OUTPUT_DIR = PROJECT_ROOT / "iff_simulator" / "output"

OWNSHIP_ID = "TARGET_1"
DEMO_IFF_CAPABILITY = "MODE_4"


def _with_demo_iff_capability(scenario: Scenario, ownship_id: str) -> Scenario:
    """Return a new Scenario with every non-Ownship aircraft marked IFF
    capable, for demonstration purposes only (see module docstring)."""
    aircraft_list = [
        aircraft
        if aircraft.aircraft_id == ownship_id
        else dataclasses.replace(aircraft, iff_capability=DEMO_IFF_CAPABILITY)
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
        scenario,
        clock,
        ownship_id=OWNSHIP_ID,
        maximum_range=2000.0,
        beam_width=360.0,
        beam_height=180.0,
        interrogation_rate=20.0,
    )
    target_selector = TargetSelector(world)
    scheduler = InterrogationScheduler(world, target_selector)

    print(f"Ownship: {OWNSHIP_ID}, interrogation period: {scheduler.period * 1000:.1f} ms")

    messages = []
    first_message = scheduler.tick()
    if first_message is not None:
        messages.append(first_message)
    while not clock.finished():
        world.step()
        message = scheduler.tick()
        if message is not None:
            messages.append(message)

    output_path = write_interrogations_csv(messages, OUTPUT_DIR / "interrogations.csv")
    print(f"Transmitted {len(messages)} interrogation(s) over {len(ownship_history)} ticks")
    print(f"Saved interrogations to {output_path}")


if __name__ == "__main__":
    main()
