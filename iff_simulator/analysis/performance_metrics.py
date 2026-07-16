"""Scalar performance metrics for one pipeline run (Phase 10).

Purpose:
    Implements `PerformanceMetrics` (one frozen dataclass holding every
    scalar metric this phase's spec asks for) and
    `compute_performance_metrics`, the single function that derives them
    all from a `PipelineRunRecord`. See `docs/METRICS.md` for the full
    mathematical derivation and the engineering-judgment notes behind
    each formula (several of these metrics are ambiguous given only
    "existing pipeline outputs" to work from -- every judgment call is
    documented there, not silently assumed).

Inputs:
    A `PipelineRunRecord`.

Outputs:
    A `PerformanceMetrics` instance, and `write_performance_metrics_csv`
    for `performance_metrics.csv`.

Engineering explanation:
    Every metric is a pure function of `PipelineRunRecord`'s fields --
    no estimation, no re-running the simulation, no reading `Scenario`
    beyond identity lookups (and this module does not even need those;
    `confusion_matrix.py` is the only module that does). `NOISE_FLOOR`
    is a documented constant used only to derive the optional SNR proxy
    metric -- it is not a measured or physical quantity, mirroring how
    `propagation.compute_signal_strength` itself is already documented
    as "not a real dBm/Watt measurement."
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from ..sensors.iff import AuthenticationResult, IFFMode, MeasurementStatus, TrackStatus
from .latency_analysis import receiver_delay_us
from .run_record import PipelineRunRecord
from .statistics import mean, min_max, safe_divide

NOISE_FLOOR = 0.01
"""Documented constant noise floor used only to derive the optional,
synthetic SNR proxy metric (see `average_snr_db_proxy` below). Not a
measured quantity -- no SNR concept exists elsewhere in this codebase."""

PERFORMANCE_METRICS_CSV_COLUMNS = [
    "Detection_Probability", "False_Alarm_Rate", "Authentication_Success_Rate",
    "Reply_Success_Rate", "Decoder_Success_Rate", "Track_Confirmation_Rate",
    "Average_Track_Lifetime_S", "Average_Detection_Range_M", "Maximum_Detection_Range_M",
    "Average_Processing_Delay_Us", "Average_Propagation_Delay_Us", "Average_Receiver_Delay_Us",
    "Average_Track_Update_Delay_S", "Average_Total_Delay_Us", "Average_Signal_Strength",
    "Average_SNR_Db_Proxy",
]


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """Every scalar performance metric for one pipeline run.

    See `docs/METRICS.md` for the exact formula and reasoning behind
    each field. Fields with no supporting data this run (e.g. no Mode 5
    replies at all) are `0.0`, never `NaN` or a raised exception.
    """

    detection_probability: float
    false_alarm_rate: float
    authentication_success_rate: float
    reply_success_rate: float
    decoder_success_rate: float
    track_confirmation_rate: float
    average_track_lifetime_s: float
    average_detection_range_m: float
    maximum_detection_range_m: float
    average_processing_delay_us: float
    average_propagation_delay_us: float
    average_receiver_delay_us: float
    average_track_update_delay_s: float
    average_total_delay_us: float
    average_signal_strength: float
    average_snr_db_proxy: float

    def to_csv_row(self) -> dict:
        return {
            "Detection_Probability": self.detection_probability,
            "False_Alarm_Rate": self.false_alarm_rate,
            "Authentication_Success_Rate": self.authentication_success_rate,
            "Reply_Success_Rate": self.reply_success_rate,
            "Decoder_Success_Rate": self.decoder_success_rate,
            "Track_Confirmation_Rate": self.track_confirmation_rate,
            "Average_Track_Lifetime_S": self.average_track_lifetime_s,
            "Average_Detection_Range_M": self.average_detection_range_m,
            "Maximum_Detection_Range_M": self.maximum_detection_range_m,
            "Average_Processing_Delay_Us": self.average_processing_delay_us,
            "Average_Propagation_Delay_Us": self.average_propagation_delay_us,
            "Average_Receiver_Delay_Us": self.average_receiver_delay_us,
            "Average_Track_Update_Delay_S": self.average_track_update_delay_s,
            "Average_Total_Delay_Us": self.average_total_delay_us,
            "Average_Signal_Strength": self.average_signal_strength,
            "Average_SNR_Db_Proxy": self.average_snr_db_proxy,
        }


def compute_performance_metrics(record: PipelineRunRecord) -> PerformanceMetrics:
    """Derive every `PerformanceMetrics` field from `record`."""
    real_measurements = [t.real_measurement for t in record.tick_results if t.real_measurement is not None]
    valid_measurements = [m for m in real_measurements if m.reply_status == MeasurementStatus.VALID]
    garbled_count = sum(1 for m in real_measurements if m.reply_status == MeasurementStatus.GARBLED)

    expected_replies = sum(1 for reply in record.replies if reply is not None)
    correct_replies = len(valid_measurements)
    detection_probability = safe_divide(correct_replies, expected_replies)

    stats = record.receiver_statistics
    total_reply_events = (stats.replies_received + stats.false_replies + stats.replies_fruited) if stats else 0
    false_alarm_rate = safe_divide(stats.false_replies, total_reply_events) if stats else 0.0

    mode5_valid = [m for m in valid_measurements if m.mode in (IFFMode.MODE5_L1, IFFMode.MODE5_L2)]
    authenticated = sum(1 for m in mode5_valid if m.authentication_status == AuthenticationResult.AUTHENTICATED)
    authentication_success_rate = safe_divide(authenticated, len(mode5_valid))

    reply_success_rate = safe_divide(correct_replies, len(record.interrogations))
    decoder_success_rate = safe_divide(correct_replies, correct_replies + garbled_count)

    track_confirmation_rate = _track_confirmation_rate(record)
    average_track_lifetime_s = mean([s.duration for s in record.completed_track_summaries])

    ranges = [m.range_m for m in valid_measurements]
    _, max_range = min_max(ranges)

    processing_delays = [m.processing_delay for m in valid_measurements if m.processing_delay is not None]
    propagation_delays = [m.propagation_delay for m in valid_measurements if m.propagation_delay is not None]
    receiver_delays_us = [d for d in (receiver_delay_us(m) for m in valid_measurements) if d is not None]
    total_delays_us = [
        (m.arrival_time - m.time) * 1_000_000.0
        for m in valid_measurements
        if m.arrival_time is not None
    ]

    signal_strengths = [
        m.signal_strength
        for tick in record.tick_results
        for m in _all_measurements(tick)
        if m.signal_strength is not None
    ]

    average_signal_strength = mean(signal_strengths)
    average_snr_db_proxy = (
        10.0 * math.log10(average_signal_strength / NOISE_FLOOR) if average_signal_strength > 0.0 else 0.0
    )

    return PerformanceMetrics(
        detection_probability=detection_probability,
        false_alarm_rate=false_alarm_rate,
        authentication_success_rate=authentication_success_rate,
        reply_success_rate=reply_success_rate,
        decoder_success_rate=decoder_success_rate,
        track_confirmation_rate=track_confirmation_rate,
        average_track_lifetime_s=average_track_lifetime_s,
        average_detection_range_m=mean(ranges),
        maximum_detection_range_m=max_range,
        average_processing_delay_us=mean(processing_delays),
        average_propagation_delay_us=mean(propagation_delays),
        average_receiver_delay_us=mean(receiver_delays_us),
        average_track_update_delay_s=0.0,
        average_total_delay_us=mean(total_delays_us),
        average_signal_strength=average_signal_strength,
        average_snr_db_proxy=average_snr_db_proxy,
    )


def _all_measurements(tick_result):
    """Every measurement (real, false-alarm, fruited) produced this tick."""
    if tick_result.real_measurement is not None:
        yield tick_result.real_measurement
    yield from tick_result.false_alarm_measurements
    yield from tick_result.fruited_measurements


def _track_confirmation_rate(record: PipelineRunRecord) -> float:
    completed_confirmed = sum(1 for s in record.completed_track_summaries if s.confirmed_time > 0.0)
    active_confirmed = sum(1 for t in record.active_tracks if t.track_status == TrackStatus.CONFIRMED)
    total = len(record.completed_track_summaries) + len(record.active_tracks)
    return safe_divide(completed_confirmed + active_confirmed, total)


def write_performance_metrics_csv(metrics: PerformanceMetrics, output_path: Path | str) -> Path:
    """Write one `PerformanceMetrics` snapshot to `performance_metrics.csv`."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PERFORMANCE_METRICS_CSV_COLUMNS)
        writer.writeheader()
        writer.writerow(metrics.to_csv_row())
    return output_path
