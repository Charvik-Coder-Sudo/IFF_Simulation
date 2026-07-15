"""Ground Truth subsystem: load, validate, merge, and inspect aircraft trajectories.

This package is the reusable foundation every future MSDF module
(Ownship, Geometry, Airborne PSR, IFF, Scheduler, Receiver, Decoder) is
built on top of. Phase 1 scope only: no IFF, Mode S, Mode 5, tracking,
or geometry logic lives here.
"""

from .inspector import GroundTruthInspector
from .loader import GroundTruthLoader
from .merger import GroundTruthMerger
from .models import REQUIRED_COLUMNS
from .statistics import GroundTruthStatistics
from .validator import GroundTruthValidationError, GroundTruthValidator

__all__ = [
    "GroundTruthInspector",
    "GroundTruthLoader",
    "GroundTruthMerger",
    "GroundTruthStatistics",
    "GroundTruthValidationError",
    "GroundTruthValidator",
    "REQUIRED_COLUMNS",
]
