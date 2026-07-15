"""Phase 2 entry point: run the runtime World model over recorded ground truth.

Purpose:
    Demonstrates the Phase 2 architecture end to end: Scenario ->
    SimulationClock -> World -> Ownship -> Targets. Loads the recorded
    ground truth, designates Target 1 as Ownship, and steps the
    simulation clock through the entire recorded time range, printing a
    validation snapshot along the way.

Inputs:
    None (paths are derived relative to this file's location).

Outputs:
    Console output only: current time, Ownship position, alive targets,
    and IFF-capable targets at intervals through the run.

Engineering explanation:
    This script contains no geometry, range, azimuth, elevation, beam,
    scheduler, Mode S, Mode 5, interrogation, or reply logic — it only
    exercises `World.step()`, which itself does no motion propagation:
    it advances simulated time and points each aircraft's current-state
    cursor at the matching already-recorded sample. Printing every one
    of the ~29,000 recorded steps would flood the console, so the
    validation snapshot is printed at a throttled interval (plus the
    first and last step) rather than every single step; the loop itself
    still steps through every recorded sample.
"""

from __future__ import annotations

from pathlib import Path

from iff_simulator.ground_truth import GroundTruthLoader
from iff_simulator.simulation import SimulationClock, World

PROJECT_ROOT = Path(__file__).resolve().parent
AIRCRAFTS_DIR = PROJECT_ROOT / "Aircrafts"

OWNSHIP_ID = "TARGET_1"
PRINT_INTERVAL = 5000


def _print_snapshot(world: World) -> None:
    ownship = world.ownship
    print(f"Current Time:       {world.current_time()}")
    print(f"Ownship Position:   {ownship.position}")
    print(f"Alive Targets:      {[a.aircraft_id for a in world.alive_targets()]}")
    print(f"IFF Capable Targets:{[a.aircraft_id for a in world.iff_capable_targets()]}")
    print("-" * 40)


def main() -> None:
    scenario = GroundTruthLoader(AIRCRAFTS_DIR).load()

    ownship_history = scenario.get_state_history(OWNSHIP_ID)
    start_time = ownship_history[0].time
    end_time = ownship_history[-1].time
    dt = ownship_history[1].time - ownship_history[0].time

    clock = SimulationClock(start_time=start_time, dt=dt, end_time=end_time)
    world = World(scenario, clock, ownship_id=OWNSHIP_ID)

    print(f"Ownship designated: {OWNSHIP_ID}")
    print(f"Targets:            {[a.aircraft_id for a in world.get_targets()]}")
    print("-" * 40)
    _print_snapshot(world)

    step_count = 0
    while not clock.finished():
        world.step()
        step_count += 1
        if step_count % PRINT_INTERVAL == 0 or clock.finished():
            _print_snapshot(world)


if __name__ == "__main__":
    main()
