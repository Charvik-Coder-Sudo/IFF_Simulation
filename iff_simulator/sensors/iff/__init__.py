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
    write_receiver_statistics_csv,
    write_replies_csv,
    write_tracks_csv,
)
from .decoder import ModeDecoder
from .detection import (
    PD_MODEL_ALWAYS_DETECT,
    PD_MODEL_GAUSSIAN,
    PD_MODEL_INVERSE_QUARTIC,
    compute_pd,
    pd_always_detect,
    pd_gaussian,
    pd_inverse_quartic,
)
from .false_replies import FalseReply, FalseReplyGenerator
from .fruiting import FruitedReply, FruitingGenerator
from .garbling import detect_garbled
from .interrogation import InterrogationMessage
from .interrogation_queue import InterrogationQueue, write_interrogations_csv
from .jitter import JitteredReplyPropagation, jitter_processing_delay
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
from .noise import apply_measurement_noise
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
from .receiver_config import ReceiverConfig
from .receiver_pipeline import ReceiverEffectsPipeline, ReceiverTickResult
from .receiver_statistics import (
    RECEIVER_STATISTICS_CSV_COLUMNS,
    ReceiverStatistics,
    ReceiverStatisticsCollector,
)
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
    "PD_MODEL_ALWAYS_DETECT",
    "PD_MODEL_GAUSSIAN",
    "PD_MODEL_INVERSE_QUARTIC",
    "RECEIVER_STATISTICS_CSV_COLUMNS",
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
    "FalseReply",
    "FalseReplyGenerator",
    "FriendFoeStatus",
    "FruitedReply",
    "FruitingGenerator",
    "IFFMeasurementReport",
    "IFFMode",
    "IFFTrack",
    "IFFTrackManager",
    "InterrogationMessage",
    "InterrogationQueue",
    "InterrogationScheduler",
    "JitteredReplyPropagation",
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
    "ReceiverConfig",
    "ReceiverEffectsPipeline",
    "ReceiverStatistics",
    "ReceiverStatisticsCollector",
    "ReceiverTickResult",
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
    "apply_measurement_noise",
    "classify_friendly_status",
    "compute_icao_address",
    "compute_mission_code",
    "compute_pd",
    "compute_signal_strength",
    "derive_authentication_status",
    "derive_friend_foe_status",
    "detect_garbled",
    "jitter_processing_delay",
    "pd_always_detect",
    "pd_gaussian",
    "pd_inverse_quartic",
    "write_decoded_csv",
    "write_interrogations_csv",
    "write_receiver_statistics_csv",
    "write_replies_csv",
    "write_track_summary_csv",
    "write_tracks_csv",
]
