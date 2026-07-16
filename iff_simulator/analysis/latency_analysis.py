"""Latency breakdown: Scheduler / Processing / Propagation / Receiver /
Track Update / Total End-to-End delay (Phase 10).

Purpose:
    Implements `compute_latency_breakdown`, which reports mean/min/max/
    stdev for each of the six delay components this phase's spec asks
    for, derived entirely from timestamps already present on
    `DecodedIFFMeasurement` -- never a new timing measurement.

Inputs:
    A `PipelineRunRecord`.

Outputs:
    A `LatencyBreakdown`, and `write_latency_statistics_csv` for
    `latency_statistics.csv`.

Engineering explanation:
    Two of the six components -- Scheduler Delay and Track Update Delay
    -- are `0.0` by construction in the current architecture, not an
    approximation this phase invents: `InterrogationScheduler.tick()`
    transmits an interrogation synchronously, in the same call that
    decided to (no queuing delay exists to measure), and
    `IFFTrackManager.update()` is always called synchronously in the
    same call stack as decoding the measurement that feeds it (no
    deferred/batched track update exists either). Receiver Delay is a
    genuine computed sanity-check quantity (`receiver_delay_us`,
    reused by `performance_metrics.py` too) -- also expected to be ~0
    given Phase 9's current propagation model, but computed for real
    from `arrival_time` so a future phase that *does* introduce
    receiver-internal latency would show up here without any change to
    this module.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from ..sensors.iff import MeasurementStatus
from .run_record import PipelineRunRecord
from .statistics import mean, min_max, population_stdev

LATENCY_STATISTICS_CSV_COLUMNS = ["Component", "Mean", "Min", "Max", "Stdev"]


def receiver_delay_us(measurement) -> float | None:
    """`arrival_time - (time + (processing_delay + propagation_delay)/1e6)`,
    microseconds. `None` if any required timestamp is missing (e.g. a
    NO_REPLY/GARBLED measurement)."""
    if measurement.arrival_time is None or measurement.processing_delay is None or measurement.propagation_delay is None:
        return None
    expected_arrival = measurement.time + (measurement.processing_delay + measurement.propagation_delay) / 1_000_000.0
    return (measurement.arrival_time - expected_arrival) * 1_000_000.0


@dataclass(frozen=True, slots=True)
class LatencyComponentStats:
    """Mean/min/max/stdev for one latency component."""

    mean: float
    minimum: float
    maximum: float
    stdev: float

    def to_csv_row(self, component_name: str) -> dict:
        return {"Component": component_name, "Mean": self.mean, "Min": self.minimum, "Max": self.maximum, "Stdev": self.stdev}


def _component_stats(values: list) -> LatencyComponentStats:
    lo, hi = min_max(values)
    return LatencyComponentStats(mean=mean(values), minimum=lo, maximum=hi, stdev=population_stdev(values))


@dataclass(frozen=True, slots=True)
class LatencyBreakdown:
    """The six-component latency breakdown this phase's spec asks for."""

    scheduler_delay: LatencyComponentStats
    processing_delay: LatencyComponentStats
    propagation_delay: LatencyComponentStats
    receiver_delay: LatencyComponentStats
    track_update_delay: LatencyComponentStats
    total_end_to_end_delay: LatencyComponentStats

    def to_csv_rows(self) -> list:
        return [
            self.scheduler_delay.to_csv_row("Scheduler_Delay_Us"),
            self.processing_delay.to_csv_row("Processing_Delay_Us"),
            self.propagation_delay.to_csv_row("Propagation_Delay_Us"),
            self.receiver_delay.to_csv_row("Receiver_Delay_Us"),
            self.track_update_delay.to_csv_row("Track_Update_Delay_Us"),
            self.total_end_to_end_delay.to_csv_row("Total_End_To_End_Delay_Us"),
        ]


def _valid_real_measurements(record: PipelineRunRecord):
    return [
        t.real_measurement
        for t in record.tick_results
        if t.real_measurement is not None and t.real_measurement.reply_status == MeasurementStatus.VALID
    ]


def compute_latency_breakdown(record: PipelineRunRecord) -> LatencyBreakdown:
    """Derive the full `LatencyBreakdown` from `record`."""
    valid = _valid_real_measurements(record)

    scheduler_delays = [0.0] * len(valid)
    track_update_delays = [0.0] * len(valid)
    processing_delays = [m.processing_delay for m in valid if m.processing_delay is not None]
    propagation_delays = [m.propagation_delay for m in valid if m.propagation_delay is not None]
    receiver_delays = [d for d in (receiver_delay_us(m) for m in valid) if d is not None]
    total_delays = [(m.arrival_time - m.time) * 1_000_000.0 for m in valid if m.arrival_time is not None]

    return LatencyBreakdown(
        scheduler_delay=_component_stats(scheduler_delays),
        processing_delay=_component_stats(processing_delays),
        propagation_delay=_component_stats(propagation_delays),
        receiver_delay=_component_stats(receiver_delays),
        track_update_delay=_component_stats(track_update_delays),
        total_end_to_end_delay=_component_stats(total_delays),
    )


def write_latency_statistics_csv(breakdown: LatencyBreakdown, output_path: Path | str) -> Path:
    """Write a `LatencyBreakdown` to `latency_statistics.csv`, one row per component."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LATENCY_STATISTICS_CSV_COLUMNS)
        writer.writeheader()
        for row in breakdown.to_csv_rows():
            writer.writerow(row)
    return output_path
