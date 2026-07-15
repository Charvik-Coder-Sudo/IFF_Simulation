"""Tests for GroundTruthValidator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from iff_simulator.ground_truth import GroundTruthLoader
from iff_simulator.ground_truth.validator import (
    GroundTruthValidationError,
    GroundTruthValidator,
)


def test_validate_passes_on_well_formed_data(aircrafts_dir: Path) -> None:
    trajectories = GroundTruthLoader(aircrafts_dir).load()
    validator = GroundTruthValidator(trajectories)
    validator.validate(verbose=False)  # should not raise


def test_validate_raises_on_missing_column(aircrafts_dir: Path) -> None:
    trajectories = GroundTruthLoader(aircrafts_dir).load()
    bad = trajectories["TARGET_1"].drop(columns=["X"])
    validator = GroundTruthValidator({"TARGET_1": bad})
    with pytest.raises(GroundTruthValidationError):
        validator.validate(verbose=False)


def test_validate_raises_on_missing_values(aircrafts_dir: Path) -> None:
    trajectories = GroundTruthLoader(aircrafts_dir).load()
    bad = trajectories["TARGET_1"].copy()
    bad.loc[0, "X"] = np.nan
    validator = GroundTruthValidator({"TARGET_1": bad})
    with pytest.raises(GroundTruthValidationError):
        validator.validate(verbose=False)


def test_validate_raises_on_non_increasing_time(aircrafts_dir: Path) -> None:
    trajectories = GroundTruthLoader(aircrafts_dir).load()
    bad = trajectories["TARGET_1"].copy()
    bad.loc[1, "Time"] = bad.loc[0, "Time"]
    validator = GroundTruthValidator({"TARGET_1": bad})
    with pytest.raises(GroundTruthValidationError):
        validator.validate(verbose=False)


def test_validate_raises_on_non_constant_timestep(aircrafts_dir: Path) -> None:
    trajectories = GroundTruthLoader(aircrafts_dir).load()
    bad = trajectories["TARGET_1"].copy()
    bad.loc[2, "Time"] = bad.loc[2, "Time"] + 5
    validator = GroundTruthValidator({"TARGET_1": bad})
    with pytest.raises(GroundTruthValidationError):
        validator.validate(verbose=False)


def test_validate_does_not_raise_on_differing_sample_counts(aircrafts_dir: Path) -> None:
    trajectories = GroundTruthLoader(aircrafts_dir).load()
    validator = GroundTruthValidator(trajectories)
    validator.validate(verbose=False)  # TARGET_1 has 5 rows, TARGET_2 has 4
