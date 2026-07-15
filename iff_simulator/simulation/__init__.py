"""Simulation scaffolding: SimulationClock and WorldState.

No motion propagation, geometry, or IFF logic lives here — only the
minimal time-advancement and current-state bookkeeping later phases
will build on.
"""

from .clock import SimulationClock
from .world import WorldState

__all__ = ["SimulationClock", "WorldState"]
