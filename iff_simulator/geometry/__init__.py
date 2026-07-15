"""Geometry package: the shared mathematical foundation for every sensor.

`GeometryEngine` computes `RelativeState` (range, azimuth, elevation,
bearing, closing velocity, relative position/velocity) between Ownship
and a target. `vector_math` provides the reusable vector operations it
is built from. No IFF, PSR, AESA, EO/IR, or Sensor Fusion logic lives
here — every future sensor module is expected to call into this
package rather than re-deriving this math.
"""

from . import vector_math
from .geometry_engine import GeometryEngine
from .relative_state import RelativeState

__all__ = ["GeometryEngine", "RelativeState", "vector_math"]
