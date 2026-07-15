"""Validates a loaded ground-truth Scenario before it is trusted downstream.

Purpose:
    Implements `GroundTruthValidator`, which runs a fixed set of
    integrity checks against every aircraft's recorded state history in
    a `Scenario` built by `GroundTruthLoader`, before any merging,
    inspection, or visualization happens.

Inputs:
    A `Scenario` as produced by `GroundTruthLoader.load()`.

Outputs:
    A printed validation report. Raises `GroundTruthValidationError` with
    a descriptive message the moment a hard-integrity check fails.

Engineering explanation:
    Checks are split into "hard" checks (missing/non-numeric values,
    non-increasing timestamps, non-constant per-aircraft timestep) which
    raise immediately, and "informational" checks (sample count
    differences across aircraft) which are only reported, because
    different aircraft legitimately fly for different durations and
    therefore produce different sample counts. Operating on `Scenario`
    domain objects instead of DataFrames means "required columns" and
    "dtypes" checks become "required attributes are present and hold
    numeric values" — the equivalent guarantee for typed domain objects.
"""

from __future__ import annotations

import math

from ..domain import AircraftState, Scenario


class GroundTruthValidationError(Exception):
    """Raised when a loaded ground-truth trajectory fails a hard integrity check."""


class GroundTruthValidator:
    """Runs integrity checks over a Scenario's recorded aircraft state histories.

    Purpose:
        Guarantee that every aircraft state history handed to the
        merger, inspector, and visualization modules is well-formed, so
        those modules never need to defend against malformed input
        themselves.

    Inputs:
        `scenario`: a `Scenario`, as produced by `GroundTruthLoader.load()`.

    Outputs:
        `validate()` prints a human-readable report and returns `None`
        on success, or raises `GroundTruthValidationError` on the first
        hard-integrity failure.

    Engineering explanation:
        Validation is deliberately fail-fast: a simulator feeding bad
        ground truth into a sensor-fusion pipeline is worse than one
        that simply refuses to run, since downstream tracking/IFF logic
        would silently produce misleading results otherwise.
    """

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario

    def validate(self, verbose: bool = True) -> None:
        """Run all validation checks, printing a report as it goes.

        Raises:
            GroundTruthValidationError: on the first hard-integrity failure.
        """
        report_lines = ["Ground Truth Validation Report", "=" * 40]

        for aircraft_id in self.scenario.list_aircraft_ids():
            history = self.scenario.get_state_history(aircraft_id)
            report_lines.append(f"\nTarget: {aircraft_id}")
            self._check_required_fields(aircraft_id, history)
            self._check_missing_values(aircraft_id, history)
            self._check_numeric_types(aircraft_id, history)
            self._check_increasing_timestamps(aircraft_id, history)
            timestep = self._check_constant_timestep(aircraft_id, history)
            report_lines.append(f"  Rows: {len(history)}")
            report_lines.append(f"  Timestep: {timestep}")
            report_lines.append("  Status: OK")

        self._report_sample_counts(report_lines)

        report_lines.append("\nAll hard integrity checks passed.")
        if verbose:
            print("\n".join(report_lines))

    def _check_required_fields(self, aircraft_id: str, history: list[AircraftState]) -> None:
        if not history:
            raise GroundTruthValidationError(f"{aircraft_id}: no recorded states")
        required_attrs = ("time", "position", "velocity", "range_m", "azimuth_deg", "elevation_deg")
        for state in history:
            missing = [attr for attr in required_attrs if getattr(state, attr, None) is None]
            if missing:
                raise GroundTruthValidationError(
                    f"{aircraft_id}: missing required fields {missing}"
                )

    def _check_missing_values(self, aircraft_id: str, history: list[AircraftState]) -> None:
        for state in history:
            values = {
                "time": state.time,
                "X": state.position.x,
                "Y": state.position.y,
                "Z": state.position.z,
                "VX": state.velocity.x,
                "VY": state.velocity.y,
                "VZ": state.velocity.z,
                "Range": state.range_m,
                "Azimuth": state.azimuth_deg,
                "Elevation": state.elevation_deg,
            }
            null_fields = [name for name, value in values.items() if _is_nan(value)]
            if null_fields:
                raise GroundTruthValidationError(
                    f"{aircraft_id}: missing values found in fields {null_fields}"
                )

    def _check_numeric_types(self, aircraft_id: str, history: list[AircraftState]) -> None:
        for state in history:
            for name, value in (
                ("time", state.time),
                ("X", state.position.x),
                ("Y", state.position.y),
                ("Z", state.position.z),
                ("VX", state.velocity.x),
                ("VY", state.velocity.y),
                ("VZ", state.velocity.z),
                ("Range", state.range_m),
                ("Azimuth", state.azimuth_deg),
                ("Elevation", state.elevation_deg),
            ):
                if not isinstance(value, (int, float)):
                    raise GroundTruthValidationError(
                        f"{aircraft_id}: field '{name}' expected a numeric value, "
                        f"got {type(value).__name__}"
                    )

    def _check_increasing_timestamps(self, aircraft_id: str, history: list[AircraftState]) -> None:
        times = [state.time for state in history]
        if any(later <= earlier for earlier, later in zip(times, times[1:])):
            raise GroundTruthValidationError(
                f"{aircraft_id}: Time is not strictly increasing"
            )

    def _check_constant_timestep(self, aircraft_id: str, history: list[AircraftState]) -> float:
        times = [state.time for state in history]
        deltas = [round(later - earlier, 6) for earlier, later in zip(times, times[1:])]
        unique_deltas = sorted(set(deltas))
        if len(unique_deltas) > 1:
            raise GroundTruthValidationError(
                f"{aircraft_id}: timestep is not constant, found values "
                f"{unique_deltas[:5]}"
            )
        return float(unique_deltas[0]) if unique_deltas else float("nan")

    def _report_sample_counts(self, report_lines: list[str]) -> None:
        counts = {
            aircraft_id: len(self.scenario.get_state_history(aircraft_id))
            for aircraft_id in self.scenario.list_aircraft_ids()
        }
        report_lines.append(f"\nSample counts per target: {counts}")
        if len(set(counts.values())) > 1:
            report_lines.append(
                "  Note: sample counts differ across targets - expected, "
                "since aircraft fly for different durations."
            )


def _is_nan(value: float) -> bool:
    try:
        return math.isnan(value)
    except TypeError:
        return False
