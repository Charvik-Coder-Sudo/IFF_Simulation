"""Simulation runtime: SimulationClock and World.

No motion propagation, geometry, or IFF logic lives here — only the
minimal time-advancement, current-state bookkeeping, and Ownship/target
query API later phases will build on.
"""

from .clock import SimulationClock
from .world import World

__all__ = ["SimulationClock", "World"]
