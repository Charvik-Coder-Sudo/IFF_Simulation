"""IFF sensor package: Ownship, target selection (Phase 4), and the
interrogation scheduler (Phase 5).

No reply, decoding, transponder, tracking, or fusion logic lives here
yet. `InterrogationScheduler` only decides when/who/what/which-mode to
interrogate; it never generates or processes a reply.
"""

from .interrogation import InterrogationMessage
from .interrogation_queue import InterrogationQueue, write_interrogations_csv
from .mode import DefaultModeSelectionPolicy, IFFMode, ModeSelectionPolicy
from .ownship import Ownship
from .scheduler import DefaultSchedulingPolicy, InterrogationScheduler, SchedulingPolicy
from .selected_target import SelectedTarget
from .selection_policy import DefaultSelectionPolicy, SelectionPolicy
from .target_selector import TargetSelector
from .uplink_format import DEFAULT_UPLINK_FORMAT_BY_MODE, UplinkFormat

__all__ = [
    "DEFAULT_UPLINK_FORMAT_BY_MODE",
    "DefaultModeSelectionPolicy",
    "DefaultSchedulingPolicy",
    "DefaultSelectionPolicy",
    "IFFMode",
    "InterrogationMessage",
    "InterrogationQueue",
    "InterrogationScheduler",
    "ModeSelectionPolicy",
    "Ownship",
    "SchedulingPolicy",
    "SelectedTarget",
    "SelectionPolicy",
    "TargetSelector",
    "UplinkFormat",
    "write_interrogations_csv",
]
