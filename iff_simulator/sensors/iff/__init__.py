"""IFF sensor package: Ownship, target selection (Phase 4), the
interrogation scheduler (Phase 5), the airborne transponder (Phase 6),
the receive/decode pipeline (Phase 7), the track manager + measurement
report generator (Phase 8), and the Phase 8.5 engineering refinements
(full relative geometry carried through the pipeline, semantic
authentication states, a deterministic signal-strength model, bounded
per-track history, completed-track summaries, and optional per-stage
CSV logging).

No sensor fusion logic lives here yet. `IFFTrackManager` maintains
logical tracks (identity, status, quality) from decoded measurements
alone — no Kalman filter, IMM, prediction, smoothing, interpolation, or
covariance.
"""

from .authentication import (
    AuthenticationEngine,
    AuthenticationResult,
    classify_friendly_status,
    derive_authentication_status,
)
from .csv_logging import (
    DECODED_CSV_COLUMNS,
    REPLIES_CSV_COLUMNS,
    TRACKS_CSV_COLUMNS,
    write_decoded_csv,
    write_replies_csv,
    write_tracks_csv,
)
from .decoder import ModeDecoder
from .interrogation import InterrogationMessage
from .interrogation_queue import InterrogationQueue, write_interrogations_csv
from .matcher import MatchResult, ReplyMatcher
from .measurement import DecodedIFFMeasurement, MeasurementStatus
from .mode import DefaultModeSelectionPolicy, IFFMode, ModeSelectionPolicy
from .mode5 import (
    MISSION_TYPES,
    Mode5Level1Payload,
    Mode5Level2Payload,
    Mode5ReplyGenerator,
    compute_mission_code,
)
from .mode_s import ModeSPayload, ModeSReplyGenerator, compute_icao_address
from .ownship import Ownship
from .propagation import (
    DEFAULT_REFERENCE_RANGE_M,
    NOMINAL_SIGNAL_STRENGTH,
    SPEED_OF_LIGHT_MPS,
    PropagatedReply,
    ReplyPropagation,
    compute_signal_strength,
)
from .receiver import TIMEOUT_SECONDS_BY_MODE, Receiver
from .receiver_buffer import ReceiverBuffer
from .reply import ReplyMessage, ReplyPayload, ReplyStatus, ReplyType
from .reply_builder import ReplyBuilder
from .report_generator import IFFMeasurementReport, ReportGenerator
from .report_writer import CSV_COLUMNS, ReportWriter
from .scheduler import DefaultSchedulingPolicy, InterrogationScheduler, SchedulingPolicy
from .selected_target import SelectedTarget
from .selection_policy import DefaultSelectionPolicy, SelectionPolicy
from .target_selector import TargetSelector
from .track import FriendFoeStatus, IFFTrack, TrackStatus, derive_friend_foe_status
from .track_manager import TRACK_HISTORY_MAXLEN, IFFTrackManager
from .track_summary import TRACK_SUMMARY_CSV_COLUMNS, TrackSummary, write_track_summary_csv
from .transponder import AirborneTransponder
from .uplink_format import DEFAULT_UPLINK_FORMAT_BY_MODE, UplinkFormat

__all__ = [
    "CSV_COLUMNS",
    "DECODED_CSV_COLUMNS",
    "DEFAULT_REFERENCE_RANGE_M",
    "DEFAULT_UPLINK_FORMAT_BY_MODE",
    "MISSION_TYPES",
    "NOMINAL_SIGNAL_STRENGTH",
    "REPLIES_CSV_COLUMNS",
    "SPEED_OF_LIGHT_MPS",
    "TIMEOUT_SECONDS_BY_MODE",
    "TRACKS_CSV_COLUMNS",
    "TRACK_HISTORY_MAXLEN",
    "TRACK_SUMMARY_CSV_COLUMNS",
    "AirborneTransponder",
    "AuthenticationEngine",
    "AuthenticationResult",
    "DecodedIFFMeasurement",
    "DefaultModeSelectionPolicy",
    "DefaultSchedulingPolicy",
    "DefaultSelectionPolicy",
    "FriendFoeStatus",
    "IFFMeasurementReport",
    "IFFMode",
    "IFFTrack",
    "IFFTrackManager",
    "InterrogationMessage",
    "InterrogationQueue",
    "InterrogationScheduler",
    "MatchResult",
    "MeasurementStatus",
    "Mode5Level1Payload",
    "Mode5Level2Payload",
    "Mode5ReplyGenerator",
    "ModeDecoder",
    "ModeSPayload",
    "ModeSReplyGenerator",
    "ModeSelectionPolicy",
    "Ownship",
    "PropagatedReply",
    "ReceiverBuffer",
    "Receiver",
    "ReplyBuilder",
    "ReplyMatcher",
    "ReplyMessage",
    "ReplyPayload",
    "ReplyPropagation",
    "ReplyStatus",
    "ReplyType",
    "ReportGenerator",
    "ReportWriter",
    "SchedulingPolicy",
    "SelectedTarget",
    "SelectionPolicy",
    "TargetSelector",
    "TrackStatus",
    "TrackSummary",
    "UplinkFormat",
    "classify_friendly_status",
    "compute_icao_address",
    "compute_mission_code",
    "compute_signal_strength",
    "derive_authentication_status",
    "derive_friend_foe_status",
    "write_decoded_csv",
    "write_interrogations_csv",
    "write_replies_csv",
    "write_track_summary_csv",
    "write_tracks_csv",
]
