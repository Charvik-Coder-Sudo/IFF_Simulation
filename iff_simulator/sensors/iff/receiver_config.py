"""Single configuration surface for every Phase 9 receiver-effects knob.

Purpose:
    Implements `ReceiverConfig` — the one dataclass that controls every
    realistic receiver effect this phase adds: probability of detection,
    false-alarm rate, sensitivity threshold, capacity/saturation, noise,
    garble window, fruiting rate, timing jitter, and the RNG seed. No
    other new module reads its own tunables from anywhere else.

Inputs:
    Plain scalars, either defaulted (every effect "off") or supplied by
    a caller (`run_receiver_pipeline.py`, tests).

Outputs:
    A `ReceiverConfig` instance, passed to `ReceiverEffectsPipeline`.

Engineering explanation:
    Every field's default value is chosen so that `ReceiverConfig()` (no
    arguments) reproduces the pre-Phase-9 "perfect logical simulator"
    behavior exactly: `pd_model="always_detect"` means Pd is always 1.0,
    `pfa`/`fruiting_rate` of 0.0 mean no false/fruited replies are ever
    injected, `sensitivity_threshold=0.0` never rejects (signal strength
    is always > 0), `capacity=None` is unbounded, every noise sigma and
    jitter bound is 0.0, and `garble_window_s=0.0` means a lone reply
    never garbles. This is what "every new feature must be optional via
    configuration" and "preserve byte-identical outputs when all new
    effects are disabled" require, satisfied by construction rather than
    by a separate on/off flag per effect.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ReceiverConfig:
    """Every tunable Phase 9 receiver-effects parameter, in one place.

    Purpose:
        Part 11's single configuration file: seed, Pd model, Pfa,
        sensitivity, capacity, noise, garble window, fruiting rate, and
        timing jitter.

    Inputs:
        Every field is independently overridable; all default to their
        "effect disabled" value (see module docstring).

    Outputs:
        Consumed by `ReceiverEffectsPipeline` and the new Pd/false-reply/
        fruiting/noise/jitter helper modules.

    Engineering explanation:
        Frozen, like every other per-instant/per-run record in this
        codebase — a run's configuration must not be mutated once a
        pipeline has been built from it.
    """

    seed: int = 0
    """RNG seed. Part 12: the same seed with the same Ground Truth and
    config must always produce identical output; a different seed must
    change receiver-level outcomes."""

    pd_model: str = "always_detect"
    """One of "always_detect" (Pd == 1.0, the default/off model),
    "gaussian" (`exp(-(R/Rmax)^2)`), or "inverse_quartic"
    (`1/(1+(R/R0)^4)`) — see `detection.py`."""

    pd_params: dict = field(default_factory=dict)
    """Parameters for `pd_model`: `{"r_max": ...}` for "gaussian",
    `{"r0": ...}` for "inverse_quartic". Unused (and may be empty) for
    "always_detect"."""

    pfa: float = 0.0
    """Probability, per tick, of injecting a false-alarm reply (Part 2).
    0.0 (default) never injects one."""

    sensitivity_threshold: float = 0.0
    """Minimum signal strength (see `propagation.compute_signal_strength`,
    range (0, 1]) a reply must meet to be decoded (Part 5). 0.0 (default)
    never rejects, since signal strength is always > 0."""

    capacity: int | None = None
    """Maximum number of replies (all origins combined) processed per
    tick (Part 6); `None` (default) is unbounded."""

    noise_sigma_range_m: float = 0.0
    """Gaussian measurement noise standard deviation for range, meters
    (Part 7). 0.0 (default) adds no noise."""

    noise_sigma_azimuth_deg: float = 0.0
    """Gaussian measurement noise standard deviation for azimuth,
    degrees (Part 7). 0.0 (default) adds no noise."""

    noise_sigma_elevation_deg: float = 0.0
    """Gaussian measurement noise standard deviation for elevation,
    degrees (Part 7). 0.0 (default) adds no noise."""

    garble_window_s: float = 0.0
    """Two replies whose arrival times differ by less than this many
    seconds are both marked GARBLED (Part 3). 0.0 (default) never
    garbles."""

    fruiting_rate: float = 0.0
    """Probability, per tick, of injecting an asynchronous fruited reply
    (Part 4). 0.0 (default) never injects one."""

    jitter_processing_delay_us: float = 0.0
    """Uniform +/- jitter bound applied to a reply's processing delay,
    microseconds (Part 8). 0.0 (default) adds no jitter."""

    jitter_propagation_delay_us: float = 0.0
    """Uniform +/- jitter bound applied to a reply's propagation delay
    (and thus its arrival time), microseconds (Part 8). 0.0 (default)
    adds no jitter."""
