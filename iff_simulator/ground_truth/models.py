"""CSV schema definition for the merged ground-truth output.

Purpose:
    Defines the canonical column name/order for `ground_truth.csv`.

Inputs:
    None directly — this module only defines a constant.

Outputs:
    `REQUIRED_COLUMNS`, imported by `GroundTruthMerger` to build the
    merged CSV in a stable, documented column order.

Engineering explanation:
    The runtime domain schema for one aircraft sample now lives in
    `iff_simulator.domain.AircraftState` (position/velocity/acceleration/
    heading/alive/time, plus the preserved raw range/azimuth/elevation
    measurement). This module only captures the *file* schema — the
    column names/order `ground_truth.csv` must use — which is a
    presentation-layer concern distinct from the runtime domain object.
"""

from __future__ import annotations

#: Canonical column names, in canonical order, for the merged ground-truth CSV.
REQUIRED_COLUMNS: list[str] = [
    "Time",
    "TargetID",
    "X",
    "Y",
    "Z",
    "VX",
    "VY",
    "VZ",
    "Range",
    "Azimuth",
    "Elevation",
]
