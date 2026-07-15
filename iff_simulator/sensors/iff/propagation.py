"""Models a reply's travel back from the target to Ownship.

Purpose:
    Implements `PropagatedReply` (the result) and `ReplyPropagation`
    (the computation), which turn a `ReplyMessage` plus the Ownship/
    target positions at reply time into an arrival time, a propagation
    delay, and a deterministic, logical-only signal strength. This is
    still not RF simulation: no pulse shape, no antenna gain pattern,
    no noise — only the light-speed transit-time delay a real reply
    would incur, and a bounded, monotonic falloff-with-range value.

Inputs:
    A `ReplyMessage`, and the Ownship/target `Vector3` positions to
    measure the return-trip distance between.

Outputs:
    A `PropagatedReply`.

Engineering explanation:
    Distance is computed via `geometry.vector_math.distance` — the
    same reusable geometry primitive every other module in this
    codebase uses — never a hand-rolled `sqrt(dx**2+dy**2+dz**2)`. This
    satisfies "reuse GeometryEngine [the geometry package]; never
    recompute geometry independently." The propagation-delay formula
    itself (distance / speed of light) is unchanged from Phase 7 —
    Phase 8.5 Part 4 only replaces the previously-constant
    `signal_strength` with the deterministic inverse-square model
    documented on `compute_signal_strength`.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...domain import Vector3
from ...geometry import vector_math
from .reply import ReplyMessage

SPEED_OF_LIGHT_MPS = 299_792_458.0
"""Speed of light in a vacuum, meters/second (exact, by SI definition)."""

_MICROSECONDS_PER_SECOND = 1_000_000.0

NOMINAL_SIGNAL_STRENGTH = 1.0
"""Retained for backward compatibility with anything that imports this
constant directly (e.g. `IFFTrackManager.update`'s default parameter
value from Phase 8) — no longer what `ReplyPropagation.propagate`
assigns; see `compute_signal_strength` for the model now used."""

DEFAULT_REFERENCE_RANGE_M = 1000.0
"""Default reference range for the signal-strength model, meters: the
range at which `compute_signal_strength` returns exactly 0.5. Chosen to
sit within this simulator's typical interrogation ranges (Ownship's
`maximum_range` has defaulted to 1-2 km across every demo script so
far), so closest/farthest targets in a typical run produce a visibly
distinct strength value rather than everything reading ~1.0 or ~0.0."""


def compute_signal_strength(range_m: float, reference_range_m: float = DEFAULT_REFERENCE_RANGE_M) -> float:
    """Deterministic, bounded, inverse-square-style signal strength.

    Purpose:
        Phase 8.5 Part 4: replace the previous fixed
        `NOMINAL_SIGNAL_STRENGTH` constant with a signal strength that
        actually varies with range — closest targets near 1, very
        distant targets near 0 — while remaining a simple, logical
        model: no antenna gain pattern, no noise, no RF simulation.

    Inputs:
        range_m: slant range, meters. Must be >= 0.
        reference_range_m: the range at which this function returns
            exactly 0.5 (dependency-injectable via
            `ReplyPropagation.__init__`; defaults to
            `DEFAULT_REFERENCE_RANGE_M`).

    Outputs:
        A float in `(0, 1]`.

    Mathematics:
        strength(r) = reference_range_m^2 / (r^2 + reference_range_m^2)

        This is a normalized inverse-square falloff:
          - strength(0) = 1.0 exactly (closest possible target).
          - strength(reference_range_m) = 0.5 (the calibration point).
          - strength(r) -> 0 as r -> infinity (very distant targets).
          - strictly monotonically decreasing in r (no local maxima/
            minima, so "closer is always stronger" holds everywhere).

    Engineering reasoning:
        Adding `reference_range_m^2` to the denominator (rather than
        dividing `K / r^2` directly) serves two purposes at once: it
        avoids a division-by-zero / infinite value at `r = 0` (so no
        epsilon fudge-factor is needed), and it makes the result
        self-normalizing into `(0, 1]` without a separate normalization
        step — the same mathematical form sometimes called a
        "Lorentzian" or half-maximum falloff curve. No randomness, no
        antenna gain, no RF waveform — a single deterministic function
        of range alone.
    """
    return (reference_range_m**2) / (range_m**2 + reference_range_m**2)


@dataclass(frozen=True, slots=True)
class PropagatedReply:
    """A ReplyMessage plus the outcome of propagating it back to Ownship.

    Purpose:
        Bundle the original reply with the facts propagation adds:
        when it arrives, how long the trip took, and its (logical)
        signal strength.

    Inputs:
        Constructed exclusively by `ReplyPropagation.propagate`.

    Outputs:
        Consumed by `ReceiverBuffer`/`Receiver`/`ReplyMatcher`.

    Engineering explanation:
        Frozen, for the same reason every other per-instant record in
        this codebase is: a fact about one reply's propagation must
        never be mutated after computation.
    """

    reply: ReplyMessage
    """The original reply this propagation result describes."""

    arrival_time: float
    """`reply.time` plus the transponder's processing delay plus the
    propagation delay (both converted from microseconds to the same
    time unit as `reply.time`)."""

    propagation_delay_us: float
    """One-way transit delay for the reply's return trip, microseconds."""

    signal_strength: float
    """Deterministic, bounded (0, 1] value from `compute_signal_strength`
    — not a real dBm/Watt measurement (see module docstring)."""


class ReplyPropagation:
    """Computes a reply's arrival time, propagation delay, and signal strength.

    Purpose:
        The single place propagation timing (and, since Phase 8.5,
        signal strength) is computed, so `Receiver`/`ReplyMatcher` never
        need to re-derive either.

    Inputs:
        speed_of_light_mps: dependency-injectable for testing; defaults
            to the real speed of light.
        reference_range_m: dependency-injectable calibration point for
            `compute_signal_strength` (Phase 8.5 Part 4); defaults to
            `DEFAULT_REFERENCE_RANGE_M`.

    Outputs:
        `propagate(reply, ownship_position, target_position) -> PropagatedReply`.

    Engineering explanation:
        Deterministic and stateless: the same
        (reply, ownship_position, target_position) triple always
        produces the same `PropagatedReply`. Signal strength reuses the
        exact same `distance_m` already computed for the propagation
        delay — one geometry computation serves both, never a second
        `vector_math.distance` call.
    """

    def __init__(
        self,
        speed_of_light_mps: float = SPEED_OF_LIGHT_MPS,
        reference_range_m: float = DEFAULT_REFERENCE_RANGE_M,
    ) -> None:
        self.speed_of_light_mps = speed_of_light_mps
        self.reference_range_m = reference_range_m

    def propagate(
        self,
        reply: ReplyMessage,
        ownship_position: Vector3,
        target_position: Vector3,
    ) -> PropagatedReply:
        """Propagate one reply back to Ownship.

        Inputs:
            reply: the `ReplyMessage` being propagated.
            ownship_position: Ownship's `Vector3` position, meters, ENU.
            target_position: the replying aircraft's `Vector3` position,
                meters, ENU (its own Ground Truth position — the same
                position the reply's geometry was already computed
                from upstream; never re-estimated here).

        Outputs:
            A `PropagatedReply`.

        Mathematics:
            distance = |target_position - ownship_position|
                       (via geometry.vector_math.distance — never
                       recomputed independently)
            propagation_delay = distance / speed_of_light   (seconds)
            arrival_time = reply.time
                           + (reply.processing_delay + propagation_delay_us)
                             / 1_000_000

        Engineering reasoning:
            `reply.processing_delay` (already stored in microseconds by
            Phase 6) represents the transponder's own turnaround time
            before transmitting; propagation delay is the return
            trip's transit time. Both are converted to the same time
            unit as `reply.time` before being added, since they are
            stored in microseconds while `reply.time` is in simulation
            time units.
        """
        distance_m = vector_math.distance(ownship_position, target_position)
        propagation_delay_s = distance_m / self.speed_of_light_mps
        propagation_delay_us = propagation_delay_s * _MICROSECONDS_PER_SECOND

        total_delay_s = (reply.processing_delay + propagation_delay_us) / _MICROSECONDS_PER_SECOND
        arrival_time = reply.time + total_delay_s

        signal_strength = compute_signal_strength(distance_m, self.reference_range_m)

        return PropagatedReply(
            reply=reply,
            arrival_time=arrival_time,
            propagation_delay_us=propagation_delay_us,
            signal_strength=signal_strength,
        )
