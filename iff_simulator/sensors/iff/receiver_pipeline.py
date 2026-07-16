"""The realistic receiver-effects orchestrator (Phase 9).

Purpose:
    Implements `ReceiverEffectsPipeline`, which replaces the "every
    valid reply arrives" assumption with the full realistic chain this
    phase specifies: Propagation -> Probability of Detection -> Receiver
    -> Garbling -> Fruiting -> Reply Loss -> Decoder, per interrogation
    per tick, plus the false-alarm injection Part 2 asks for. It composes
    the existing, unmodified `ReplyPropagation` (via the additive
    `JitteredReplyPropagation` subclass), `Receiver`, and `ModeDecoder` --
    no completed module's behavior changes.

Inputs:
    Per tick: the `InterrogationMessage` (or `None`) and the
    transponder's `ReplyMessage` (or `None`) for it, the Ownship/target
    positions, and the current simulation time.

Outputs:
    A `ReceiverTickResult`: exactly one `real_measurement`
    (`DecodedIFFMeasurement`, VALID/NO_REPLY/GARBLED) whenever an
    interrogation was given this tick, plus any false-alarm and fruited
    measurements generated this tick.

Engineering explanation:
    `ReplyMatcher` is deliberately not reused here: its identity-check is
    specific to a single real interrogation<->reply pairing, which
    false/fruited replies do not have. Instead this class builds
    `MatchResult` directly (a plain, public, frozen dataclass) -- the
    exact same fields `ReplyMatcher.match()` would have produced -- so
    the real-reply decode path is byte-identical to today's pipeline
    whenever every new effect is left at its default (off) value. See
    `test_receiver_pipeline.py`'s backward-compatibility regression test.

    Garbling requires comparing arrival times across real/false/fruited
    replies that land within a few microseconds of each other -- far
    smaller than a typical simulation `dt` (whole seconds). Rather than
    waiting for the caller's clock to advance (which would defer a
    reply's decode to a *later* tick, breaking the existing "exactly one
    measurement per interrogation, decided this tick" invariant), the
    shared `Receiver` buffer is drained each tick up to
    `current_time + drain_horizon_s` -- comfortably larger than any
    reply timeout window (microseconds) or configured timing jitter, yet
    far smaller than `dt`, so nothing is left over for the next tick in
    ordinary operation while still genuinely exercising `Receiver`'s
    arrival-ordering machinery for the garbling/saturation comparison.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ...domain import Vector3
from .authentication import AuthenticationResult
from .detection import compute_pd
from .false_replies import FalseReplyGenerator
from .garbling import detect_garbled
from .fruiting import FruitingGenerator
from .interrogation import InterrogationMessage
from .jitter import JitteredReplyPropagation, jitter_processing_delay
from .matcher import MatchResult
from .measurement import DecodedIFFMeasurement, MeasurementStatus
from .mode import IFFMode
from .noise import apply_measurement_noise
from .receiver import Receiver
from .receiver_config import ReceiverConfig
from .receiver_statistics import ReceiverStatisticsCollector
from .reply import ReplyMessage
from .decoder import ModeDecoder

DEFAULT_DRAIN_HORIZON_S = 0.01
"""Default per-tick buffer-drain horizon, seconds (see module docstring)."""


@dataclass(frozen=True, slots=True)
class _PendingOrigin:
    """Internal bookkeeping for one propagated reply awaiting decode --
    never exposed outside `ReceiverEffectsPipeline`."""

    kind: str
    """One of "REAL", "FALSE", "FRUITED"."""

    interrogation: InterrogationMessage | None
    """Set only for kind == "REAL"."""

    range_m: float
    azimuth_deg: float
    elevation_deg: float
    target_id: str


@dataclass(frozen=True, slots=True)
class ReceiverTickResult:
    """One tick's worth of decoded output from `ReceiverEffectsPipeline`."""

    real_measurement: DecodedIFFMeasurement | None
    """Exactly one measurement (VALID/NO_REPLY/GARBLED) whenever an
    interrogation was given this tick; `None` if no interrogation was
    issued this tick at all."""

    false_alarm_measurements: list
    """VALID-status measurements for phantom target_ids: freshly
    generated false alarms plus continued NO_REPLY aging for previously
    fired false alarms not yet resolved (see `mark_phantom_resolved`)."""

    fruited_measurements: list
    """FRUITED-status measurements this tick. Never track-worthy --
    callers should not feed these to `IFFTrackManager`."""


class ReceiverEffectsPipeline:
    """Orchestrates Pd, false alarms, garbling, fruiting, sensitivity,
    saturation, noise, and timing jitter around the existing receive
    chain (Phase 9).

    Inputs:
        config: a `ReceiverConfig` (defaults to `ReceiverConfig()`, every
            effect off).
        ownship_id: the interrogating Ownship's aircraft_id.
        maximum_range_m: the sensor's own configured maximum range, used
            as the synthetic geometry envelope for false/fruited replies.
        drain_horizon_s: see module docstring.

    Outputs:
        `process_tick(...) -> ReceiverTickResult`.
    """

    def __init__(
        self,
        config: ReceiverConfig | None = None,
        ownship_id: str = "OWNSHIP",
        maximum_range_m: float = 5000.0,
        drain_horizon_s: float = DEFAULT_DRAIN_HORIZON_S,
    ) -> None:
        self.config = config or ReceiverConfig()
        self.ownship_id = ownship_id
        self.maximum_range_m = maximum_range_m
        self.drain_horizon_s = drain_horizon_s

        self.rng = random.Random(self.config.seed)
        self.propagation = JitteredReplyPropagation(
            self.rng, jitter_us=self.config.jitter_propagation_delay_us
        )
        self.receiver = Receiver()
        self.decoder = ModeDecoder()
        self.false_reply_generator = FalseReplyGenerator(self.rng, ownship_id, maximum_range_m)
        self.fruiting_generator = FruitingGenerator(self.rng, ownship_id, maximum_range_m)
        self.statistics = ReceiverStatisticsCollector()

        self._pending: dict[int, _PendingOrigin] = {}
        self._phantom_ids: set[str] = set()
        self._phantom_seq_counter = 0

    def mark_phantom_resolved(self, target_id: str) -> None:
        """Tell the pipeline a phantom (false-alarm) track has been lost
        by `IFFTrackManager`, so it stops generating aging measurements
        for it."""
        self._phantom_ids.discard(target_id)

    def active_phantom_ids(self) -> list[str]:
        """Return every phantom target_id still being aged, sorted."""
        return sorted(self._phantom_ids)

    def process_tick(
        self,
        interrogation: InterrogationMessage | None,
        reply: ReplyMessage | None,
        ownship_position: Vector3,
        target_position: Vector3,
        current_time: float,
    ) -> ReceiverTickResult:
        """Process one World tick through the full realistic receive chain."""
        if interrogation is not None:
            self._submit_real(interrogation, reply, ownship_position, target_position)

        if self.config.pfa > 0.0 and self.rng.random() < self.config.pfa:
            self._submit_false(current_time, ownship_position)

        if self.config.fruiting_rate > 0.0 and self.rng.random() < self.config.fruiting_rate:
            self._submit_fruited(current_time, ownship_position)

        batch = self.receiver.pop_ready(current_time + self.drain_horizon_s)
        batch = self._apply_saturation(batch)
        garbled_ids = detect_garbled(batch, self.config.garble_window_s)

        real_measurement: DecodedIFFMeasurement | None = None
        false_alarm_measurements: list[DecodedIFFMeasurement] = []
        fruited_measurements: list[DecodedIFFMeasurement] = []

        for propagated in batch:
            origin = self._pending.pop(id(propagated))
            if id(propagated) in garbled_ids:
                self.statistics.record_garbled(current_time)
                if origin.kind == "REAL":
                    real_measurement = self._garbled_measurement(origin.interrogation)
                continue

            if origin.kind == "REAL":
                real_measurement = self._decode_real(origin.interrogation, propagated)
            elif origin.kind == "FALSE":
                measurement = self._decode_false(origin, propagated)
                false_alarm_measurements.append(measurement)
                self._phantom_ids.add(origin.target_id)
                self.statistics.record_false_reply(current_time)
            else:  # FRUITED
                measurement = self._decode_fruited(origin, propagated)
                fruited_measurements.append(measurement)
                self.statistics.record_fruited(current_time)

        if interrogation is not None and real_measurement is None:
            real_measurement = self._no_reply_measurement(interrogation)

        if interrogation is not None:
            if real_measurement.reply_status == MeasurementStatus.VALID:
                self.statistics.record_received()
            elif real_measurement.reply_status != MeasurementStatus.GARBLED:
                self.statistics.record_lost()

        fresh_phantom_ids = {m.target_id for m in false_alarm_measurements}
        for phantom_id in sorted(self._phantom_ids - fresh_phantom_ids):
            false_alarm_measurements.append(self._phantom_aging_measurement(phantom_id, current_time))

        self.statistics.record_tick_load(current_time, len(batch))

        if self._noise_configured():
            if real_measurement is not None and real_measurement.reply_status == MeasurementStatus.VALID:
                real_measurement = self._apply_noise(real_measurement)
            false_alarm_measurements = [
                self._apply_noise(m) if m.reply_status == MeasurementStatus.VALID else m
                for m in false_alarm_measurements
            ]

        return ReceiverTickResult(real_measurement, false_alarm_measurements, fruited_measurements)

    # ------------------------------------------------------------------
    # Submission (Propagation + Pd + Sensitivity, per origin)
    # ------------------------------------------------------------------

    def _submit_real(
        self,
        interrogation: InterrogationMessage,
        reply: ReplyMessage | None,
        ownship_position: Vector3,
        target_position: Vector3,
    ) -> None:
        if reply is None:
            return

        pd = compute_pd(interrogation.range_m, self.config.pd_model, self.config.pd_params)
        self.statistics.record_pd_roll(pd)
        if self.rng.random() >= pd:
            return  # not detected this tick -> resolves to NO_REPLY

        jittered_reply = jitter_processing_delay(reply, self.config.jitter_processing_delay_us, self.rng)
        propagated = self.propagation.propagate(jittered_reply, ownship_position, target_position)

        if propagated.signal_strength < self.config.sensitivity_threshold:
            return  # below sensitivity -> resolves to NO_REPLY

        self.statistics.record_signal_strength(propagated.signal_strength)
        self.statistics.record_delay(jittered_reply.processing_delay + propagated.propagation_delay_us)
        self.receiver.receive(propagated)
        self._pending[id(propagated)] = _PendingOrigin(
            kind="REAL",
            interrogation=interrogation,
            range_m=interrogation.range_m,
            azimuth_deg=interrogation.azimuth_deg,
            elevation_deg=interrogation.elevation_deg,
            target_id=interrogation.target_id,
        )

    def _submit_false(self, current_time: float, ownship_position: Vector3) -> None:
        false_reply = self.false_reply_generator.generate(current_time)
        propagated = self.propagation.propagate(false_reply.reply, ownship_position, false_reply.target_position)
        self.statistics.record_signal_strength(propagated.signal_strength)
        self.statistics.record_delay(false_reply.reply.processing_delay + propagated.propagation_delay_us)
        self.receiver.receive(propagated)
        self._pending[id(propagated)] = _PendingOrigin(
            kind="FALSE",
            interrogation=None,
            range_m=false_reply.range_m,
            azimuth_deg=false_reply.azimuth_deg,
            elevation_deg=false_reply.elevation_deg,
            target_id=false_reply.reply.target_id,
        )

    def _submit_fruited(self, current_time: float, ownship_position: Vector3) -> None:
        fruited_reply = self.fruiting_generator.generate(current_time)
        propagated = self.propagation.propagate(
            fruited_reply.reply, ownship_position, fruited_reply.target_position
        )
        self.statistics.record_signal_strength(propagated.signal_strength)
        self.statistics.record_delay(fruited_reply.reply.processing_delay + propagated.propagation_delay_us)
        self.receiver.receive(propagated)
        self._pending[id(propagated)] = _PendingOrigin(
            kind="FRUITED",
            interrogation=None,
            range_m=fruited_reply.range_m,
            azimuth_deg=fruited_reply.azimuth_deg,
            elevation_deg=fruited_reply.elevation_deg,
            target_id=fruited_reply.reply.target_id,
        )

    # ------------------------------------------------------------------
    # Saturation + Garbling (Parts 6, 3)
    # ------------------------------------------------------------------

    def _apply_saturation(self, batch: list) -> list:
        if self.config.capacity is None or len(batch) <= self.config.capacity:
            return batch
        kept, dropped = batch[: self.config.capacity], batch[self.config.capacity :]
        for propagated in dropped:
            self._pending.pop(id(propagated), None)
            self.statistics.record_lost()
        return kept

    # ------------------------------------------------------------------
    # Decode (Part 5's sensitivity is applied during submission above;
    # this is the Decoder stage of the pipeline diagram)
    # ------------------------------------------------------------------

    def _decode_real(self, interrogation: InterrogationMessage, propagated) -> DecodedIFFMeasurement:
        if self.receiver.is_timed_out(interrogation, propagated.arrival_time):
            match_result = MatchResult(interrogation=interrogation, propagated_reply=None, timed_out=True)
        else:
            match_result = MatchResult(interrogation=interrogation, propagated_reply=propagated, timed_out=False)
        return self.decoder.decode(match_result)

    def _no_reply_measurement(self, interrogation: InterrogationMessage) -> DecodedIFFMeasurement:
        match_result = MatchResult(interrogation=interrogation, propagated_reply=None, timed_out=True)
        return self.decoder.decode(match_result)

    def _garbled_measurement(self, interrogation: InterrogationMessage) -> DecodedIFFMeasurement:
        """A garbled reply is by definition unattributable to a specific
        aircraft's identity -- built directly (mirroring the shape of
        `ModeDecoder`'s own NO_REPLY branch) rather than through
        `ModeDecoder`, which has no "garbled" input concept."""
        return DecodedIFFMeasurement(
            measurement_id=interrogation.sequence_number,
            time=interrogation.time,
            target_id=interrogation.target_id,
            ownship_id=interrogation.ownship_id,
            mode=interrogation.mode,
            range_m=interrogation.range_m,
            azimuth_deg=interrogation.azimuth_deg,
            elevation_deg=interrogation.elevation_deg,
            icao_address=None,
            authentication_result=False,
            identity="UNKNOWN",
            mission=None,
            reply_status=MeasurementStatus.GARBLED,
            processing_delay=None,
            propagation_delay=None,
            arrival_time=None,
            sequence_number=interrogation.sequence_number,
            closing_velocity_mps=interrogation.closing_velocity_mps,
            relative_velocity=interrogation.relative_velocity,
            authentication_status=AuthenticationResult.NOT_APPLICABLE,
        )

    def _decode_false(self, origin: _PendingOrigin, propagated) -> DecodedIFFMeasurement:
        """VALID-status, per spec Part 2 -- so `IFFTrackManager` naturally
        starts a Tentative track for this phantom target_id."""
        reply = propagated.reply
        return DecodedIFFMeasurement(
            measurement_id=reply.reply_id,
            time=reply.time,
            target_id=origin.target_id,
            ownship_id=reply.ownship_id,
            mode=reply.mode,
            range_m=origin.range_m,
            azimuth_deg=origin.azimuth_deg,
            elevation_deg=origin.elevation_deg,
            icao_address=reply.mode_s_address,
            authentication_result=False,
            identity="UNKNOWN",
            mission=None,
            reply_status=MeasurementStatus.VALID,
            processing_delay=reply.processing_delay,
            propagation_delay=propagated.propagation_delay_us,
            arrival_time=propagated.arrival_time,
            sequence_number=reply.reply_id,
            authentication_status=AuthenticationResult.NOT_APPLICABLE,
            signal_strength=propagated.signal_strength,
        )

    def _decode_fruited(self, origin: _PendingOrigin, propagated) -> DecodedIFFMeasurement:
        """FRUITED-status -- never fed to `IFFTrackManager` by convention
        (see `ReceiverTickResult.fruited_measurements`)."""
        reply = propagated.reply
        return DecodedIFFMeasurement(
            measurement_id=reply.reply_id,
            time=reply.time,
            target_id=origin.target_id,
            ownship_id=reply.ownship_id,
            mode=reply.mode,
            range_m=origin.range_m,
            azimuth_deg=origin.azimuth_deg,
            elevation_deg=origin.elevation_deg,
            icao_address=reply.mode_s_address,
            authentication_result=False,
            identity="UNKNOWN",
            mission=None,
            reply_status=MeasurementStatus.FRUITED,
            processing_delay=reply.processing_delay,
            propagation_delay=propagated.propagation_delay_us,
            arrival_time=propagated.arrival_time,
            sequence_number=reply.reply_id,
            authentication_status=AuthenticationResult.NOT_APPLICABLE,
            signal_strength=propagated.signal_strength,
        )

    def _phantom_aging_measurement(self, target_id: str, current_time: float) -> DecodedIFFMeasurement:
        """A NO_REPLY measurement for a previously-fired false alarm that
        did not fire again this tick -- keeps `IFFTrackManager` aging its
        miss_count toward Lost, exactly like a real intermittent target."""
        self._phantom_seq_counter += 1
        return DecodedIFFMeasurement(
            measurement_id=self._phantom_seq_counter,
            time=current_time,
            target_id=target_id,
            ownship_id=self.ownship_id,
            mode=IFFMode.MODE_S,
            range_m=0.0,
            azimuth_deg=0.0,
            elevation_deg=0.0,
            icao_address=None,
            authentication_result=False,
            identity="UNKNOWN",
            mission=None,
            reply_status=MeasurementStatus.NO_REPLY,
            processing_delay=None,
            propagation_delay=None,
            arrival_time=None,
            sequence_number=self._phantom_seq_counter,
        )

    # ------------------------------------------------------------------
    # Noise (Part 7)
    # ------------------------------------------------------------------

    def _noise_configured(self) -> bool:
        return bool(
            self.config.noise_sigma_range_m
            or self.config.noise_sigma_azimuth_deg
            or self.config.noise_sigma_elevation_deg
        )

    def _apply_noise(self, measurement: DecodedIFFMeasurement) -> DecodedIFFMeasurement:
        return apply_measurement_noise(
            measurement,
            self.config.noise_sigma_range_m,
            self.config.noise_sigma_azimuth_deg,
            self.config.noise_sigma_elevation_deg,
            self.rng,
        )
