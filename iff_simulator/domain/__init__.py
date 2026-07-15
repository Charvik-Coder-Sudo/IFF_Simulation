"""Domain model: Vector3, Aircraft, AircraftState, Scenario.

These are the plain domain objects that runtime modules pass between
each other instead of pandas DataFrames. Pandas remains an
implementation detail confined to file I/O boundaries (loading .tdf
files, writing CSV output).
"""

from .aircraft import Aircraft
from .aircraft_state import AircraftState
from .scenario import Scenario
from .vector3 import Vector3

__all__ = ["Aircraft", "AircraftState", "Scenario", "Vector3"]
