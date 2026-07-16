"""Phase 10: read-only performance and detection analysis for the IFF
Mark XII pipeline.

Consumes existing outputs only (`DecodedIFFMeasurement`, `IFFTrack`,
`TrackSummary`, `ReceiverStatistics`, `ReplyMessage`,
`InterrogationMessage`, and `Scenario` for Ground Truth identity
lookups) -- no Geometry Engine, Receiver, Propagation, Scheduler, Track
Manager, Decoder, or Ground Truth module is modified by this package.
"""

from .confusion_matrix import (
    AUTHENTICATION_LABELS,
    CONFUSION_MATRIX_CSV_COLUMNS,
    IDENTITY_LABELS,
    ConfusionMatrix,
    compute_authentication_confusion_matrix,
    compute_confusion_matrix,
    compute_identity_confusion_matrix,
    write_confusion_matrix_csv,
)
from .latency_analysis import (
    LATENCY_STATISTICS_CSV_COLUMNS,
    LatencyBreakdown,
    LatencyComponentStats,
    compute_latency_breakdown,
    receiver_delay_us,
    write_latency_statistics_csv,
)
from .performance_metrics import (
    NOISE_FLOOR,
    PERFORMANCE_METRICS_CSV_COLUMNS,
    PerformanceMetrics,
    compute_performance_metrics,
    write_performance_metrics_csv,
)
from .plots import AnalysisPlotter
from .report_generator import AnalysisReportGenerator
from .roc_analysis import DEFAULT_ROC_POINTS, ROC_CURVE_CSV_COLUMNS, RocCurve, compute_roc_curve
from .run_record import PipelineRunRecord
from .statistics import (
    AUTHENTICATION_STATISTICS_CSV_COLUMNS,
    DETECTION_STATISTICS_CSV_COLUMNS,
    TRACK_STATISTICS_CSV_COLUMNS,
    AuthenticationStatistics,
    DetectionStatistics,
    TrackStatistics,
    compute_authentication_statistics,
    compute_detection_statistics,
    compute_track_statistics,
    mean,
    min_max,
    population_stdev,
    safe_divide,
    write_authentication_statistics_csv,
    write_detection_statistics_csv,
    write_track_statistics_csv,
)

__all__ = [
    "AUTHENTICATION_LABELS",
    "AUTHENTICATION_STATISTICS_CSV_COLUMNS",
    "CONFUSION_MATRIX_CSV_COLUMNS",
    "DEFAULT_ROC_POINTS",
    "DETECTION_STATISTICS_CSV_COLUMNS",
    "IDENTITY_LABELS",
    "LATENCY_STATISTICS_CSV_COLUMNS",
    "NOISE_FLOOR",
    "PERFORMANCE_METRICS_CSV_COLUMNS",
    "ROC_CURVE_CSV_COLUMNS",
    "TRACK_STATISTICS_CSV_COLUMNS",
    "AnalysisPlotter",
    "AnalysisReportGenerator",
    "AuthenticationStatistics",
    "ConfusionMatrix",
    "DetectionStatistics",
    "LatencyBreakdown",
    "LatencyComponentStats",
    "PerformanceMetrics",
    "PipelineRunRecord",
    "RocCurve",
    "TrackStatistics",
    "compute_authentication_confusion_matrix",
    "compute_authentication_statistics",
    "compute_confusion_matrix",
    "compute_detection_statistics",
    "compute_identity_confusion_matrix",
    "compute_latency_breakdown",
    "compute_performance_metrics",
    "compute_roc_curve",
    "compute_track_statistics",
    "mean",
    "min_max",
    "population_stdev",
    "receiver_delay_us",
    "safe_divide",
    "write_authentication_statistics_csv",
    "write_confusion_matrix_csv",
    "write_detection_statistics_csv",
    "write_latency_statistics_csv",
    "write_performance_metrics_csv",
    "write_track_statistics_csv",
]
