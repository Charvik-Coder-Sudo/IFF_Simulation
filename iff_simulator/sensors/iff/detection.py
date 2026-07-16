"""Probability-of-detection models, Pd(range) (Phase 9 Part 1).

Purpose:
    Implements the handful of deterministic Pd(range) curves a receiver
    can be configured with, plus `compute_pd`, the single dispatcher
    `ReceiverEffectsPipeline` calls to turn `(range_m, ReceiverConfig)`
    into a probability the caller then rolls a seeded RNG draw against.

Inputs:
    A slant range (meters) and a `ReceiverConfig.pd_model` +
    `ReceiverConfig.pd_params`.

Outputs:
    A float in `[0, 1]`.

Engineering explanation:
    Every model here is a pure, deterministic function of range alone —
    the randomness this phase introduces lives entirely in the caller's
    RNG draw against the returned probability, never inside these
    functions. `pd_always_detect` (Pd == 1.0 for every range) is the
    default model, chosen specifically so a `ReceiverConfig()` with no
    overrides reproduces the pre-Phase-9 "every valid reply arrives"
    behavior exactly.
"""

from __future__ import annotations

import math

PD_MODEL_ALWAYS_DETECT = "always_detect"
PD_MODEL_GAUSSIAN = "gaussian"
PD_MODEL_INVERSE_QUARTIC = "inverse_quartic"


def pd_always_detect(range_m: float) -> float:
    """Constant Pd == 1.0, regardless of range — the default/off model."""
    return 1.0


def pd_gaussian(range_m: float, r_max: float) -> float:
    """Gaussian-falloff detection probability.

    Mathematics:
        Pd(r) = exp(-(r / r_max)^2)

    Inputs:
        range_m: slant range, meters. Must be >= 0.
        r_max: the range at which Pd has fallen to `exp(-1) ~= 0.368`.
            Must be > 0.

    Outputs:
        A float in `(0, 1]`: 1.0 at r=0, monotonically decreasing,
        approaching (never reaching) 0 as r -> infinity.
    """
    return math.exp(-((range_m / r_max) ** 2))


def pd_inverse_quartic(range_m: float, r0: float) -> float:
    """Inverse-quartic-falloff detection probability.

    Mathematics:
        Pd(r) = 1 / (1 + (r / r0)^4)

    Inputs:
        range_m: slant range, meters. Must be >= 0.
        r0: the range at which Pd == 0.5. Must be > 0.

    Outputs:
        A float in `(0, 1]`: 1.0 at r=0, monotonically decreasing,
        approaching (never reaching) 0 as r -> infinity.
    """
    return 1.0 / (1.0 + (range_m / r0) ** 4)


_PD_MODELS = {
    PD_MODEL_ALWAYS_DETECT: lambda range_m, params: pd_always_detect(range_m),
    PD_MODEL_GAUSSIAN: lambda range_m, params: pd_gaussian(range_m, params["r_max"]),
    PD_MODEL_INVERSE_QUARTIC: lambda range_m, params: pd_inverse_quartic(range_m, params["r0"]),
}


def compute_pd(range_m: float, pd_model: str, pd_params: dict) -> float:
    """Dispatch to the configured Pd(range) model and clamp to [0, 1].

    Purpose:
        The single place `ReceiverEffectsPipeline` (or any other caller)
        turns a `ReceiverConfig`'s `pd_model`/`pd_params` into an actual
        probability, without needing to know which formula is behind it.

    Inputs:
        range_m: slant range, meters.
        pd_model: one of `PD_MODEL_ALWAYS_DETECT`, `PD_MODEL_GAUSSIAN`,
            `PD_MODEL_INVERSE_QUARTIC`.
        pd_params: the model's parameters (see each function's docstring).

    Outputs:
        A float in `[0, 1]`.

    Raises:
        ValueError: if `pd_model` is not one of the known models.

    Engineering reasoning:
        Clamping defensively (rather than trusting the formula) keeps
        the "0 <= Pd <= 1" invariant true even for a pathological
        `pd_params` value, without adding a validation step every caller
        would otherwise have to remember to run first.
    """
    if pd_model not in _PD_MODELS:
        raise ValueError(f"Unknown pd_model: {pd_model!r}")
    pd = _PD_MODELS[pd_model](range_m, pd_params)
    return max(0.0, min(1.0, pd))
