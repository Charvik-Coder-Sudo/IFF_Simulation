"""Maintains persistent logical IFF tracks from decoded measurements.

Purpose:
    Implements `IFFTrackManager`: allocates a Track_ID the first time an
    aircraft is seen, confirms a track after 3 consecutive valid replies,
    updates a confirmed/tentative track's geometry and identity fields on
    each valid reply, counts misses on NO_REPLY, and deletes (loses) a
    track once its miss count reaches a configurable threshold. This is
    track *management* only — no motion estimation, no filtering, no
    prediction, no covariance, no fusion.

Inputs:
    One `DecodedIFFMeasurement` (Phase 7, extended by Phase 8.5 Part 1
    with `relative_velocity`/`closing_velocity_mps` and Part 4 with
    `signal_strength`) per call to `update`. Optional caller-supplied
    overrides (`relative_velocity`, `reply_type`, `signal_strength`)
    remain for backward compatibility with Phase 8 callers, but a
    caller that omits them now gets the real values straight from the
    measurement — no second `GeometryEngine`/`ReplyPropagation` call
    needed anywhere in this class.

Outputs:
    `IFFTrack` snapshots (immutable) via `update`, `get_track`, and
    `get_active_tracks`; a bounded 20-item history per track via
    `get_track_history` (Phase 8.5 Part 5); full-lifetime
    `TrackSummary` records for completed (lost) tracks via
    `get_completed_track_summaries` (Phase 8.5 Part 6).

Engineering explanation:
    Association is by `aircraft_id` alone — a plain dict keyed lookup,
    never nearest-neighbor, Mahalanobis distance, the Hungarian
    algorithm, JPDA, or MHT. This is the entire "association" step this
    phase requires: exactly one aircraft_id maps to exactly one track.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from ...domain import Vector3
from .authentication import AuthenticationResult
from .measurement import DecodedIFFMeasurement, MeasurementStatus
from .mode import IFFMode
from .propagation import NOMINAL_SIGNAL_STRENGTH
from .reply import ReplyType
from .track import FriendFoeStatus, IFFTrack, TrackStatus, derive_friend_foe_status
from .track_summary import TrackSummary

TRACK_HISTORY_MAXLEN = 20
"""Phase 8.5 Part 5: bounded number of previous IFFTrack snapshots each
active track retains, for debugging/visualization/statistics only —
never used for prediction, smoothing, or filtering."""


@dataclass(slots=True)
class _TrackState:
    """Internal mutable per-aircraft bookkeeping — never exposed directly.

    Engineering explanation:
        This is the *only* mutable state in the whole track/report
        subsystem, and it is exactly what a track manager is for: it is
        not motion estimation, not a filter, not a covariance — just the
        bookkeeping described in this phase (current field values,
        miss_count, confirmation streak, bounded history, and
        full-lifetime summary accumulators).
    """

    track_id: int
    aircraft_id: str
    ownship_id: str
    time: float
    range_m: float
    azimuth_deg: float
    elevation_deg: float
    relative_velocity: Vector3 | None
    closing_velocity_mps: float | None
    mode: IFFMode
    reply_status: MeasurementStatus
    mode_s_address: str | None
    authentication_result: bool
    authentication_status: AuthenticationResult
    friend_foe_status: FriendFoeStatus
    track_status: TrackStatus
    track_quality: float
    last_update_time: float
    sequence_number: int
    reply_type: ReplyType | None
    signal_strength: float
    propagation_delay: float | None
    miss_count: int = 0
    consecutive_valid_count: int = 0
    history: deque = field(default_factory=lambda: deque(maxlen=TRACK_HISTORY_MAXLEN))

    # --- Phase 8.5 Part 6: full-lifetime summary accumulators ---
    start_time: float = 0.0
    replies_received: int = 0
    replies_missed: int = 0
    tentative_time: float = 0.0
    confirmed_time: float = 0.0
    lost_time: float = 0.0
    range_sum: float = 0.0
    range_count: int = 0
    range_min: float | None = None
    range_max: float | None = None
    signal_strength_sum: float = 0.0
    signal_strength_count: int = 0
    signal_strength_max: float | None = None

    def snapshot(self) -> IFFTrack:
        return IFFTrack(
            track_id=self.track_id,
            aircraft_id=self.aircraft_id,
            ownship_id=self.ownship_id,
            time=self.time,
            range_m=self.range_m,
            azimuth_deg=self.azimuth_deg,
            elevation_deg=self.elevation_deg,
            relative_velocity=self.relative_velocity,
            mode=self.mode,
            reply_status=self.reply_status,
            mode_s_address=self.mode_s_address,
            authentication_result=self.authentication_result,
            friend_foe_status=self.friend_foe_status,
            track_status=self.track_status,
            track_quality=self.track_quality,
            last_update_time=self.last_update_time,
            sequence_number=self.sequence_number,
            reply_type=self.reply_type,
            confidence=self.track_quality,
            signal_strength=self.signal_strength,
            propagation_delay=self.propagation_delay,
            closing_velocity_mps=self.closing_velocity_mps,
            authentication_status=self.authentication_status,
        )

    def record_range_and_signal(self, range_m: float, signal_strength: float) -> None:
        """Update the running range/signal-strength aggregates (valid replies only)."""
        self.range_sum += range_m
        self.range_count += 1
        self.range_min = range_m if self.range_min is None else min(self.range_min, range_m)
        self.range_max = range_m if self.range_max is None else max(self.range_max, range_m)

        self.signal_strength_sum += signal_strength
        self.signal_strength_count += 1
        self.signal_strength_max = (
            signal_strength if self.signal_strength_max is None else max(self.signal_strength_max, signal_strength)
        )

    def accumulate_status_time(self, new_time: float) -> None:
        """Credit the elapsed time since the last update to whichever
        status was active going into this update (Phase 8.5 Part 6)."""
        delta = new_time - self.time
        if delta <= 0:
            return
        if self.track_status == TrackStatus.TENTATIVE:
            self.tentative_time += delta
        elif self.track_status == TrackStatus.CONFIRMED:
            self.confirmed_time += delta
        elif self.track_status == TrackStatus.LOST:
            self.lost_time += delta

    def build_summary(self) -> TrackSummary:
        """Build this track's full-lifetime TrackSummary (called once, at loss)."""
        avg_range = self.range_sum / self.range_count if self.range_count else 0.0
        avg_signal_strength = (
            self.signal_strength_sum / self.signal_strength_count if self.signal_strength_count else 0.0
        )
        return TrackSummary(
            track_id=self.track_id,
            aircraft_id=self.aircraft_id,
            ownship_id=self.ownship_id,
            track_start_time=self.start_time,
            track_end_time=self.time,
            duration=self.time - self.start_time,
            replies_received=self.replies_received,
            replies_missed=self.replies_missed,
            tentative_time=self.tentative_time,
            confirmed_time=self.confirmed_time,
            lost_time=self.lost_time,
            max_range_m=self.range_max if self.range_max is not None else 0.0,
            min_range_m=self.range_min if self.range_min is not None else 0.0,
            avg_range_m=avg_range,
            max_signal_strength=self.signal_strength_max if self.signal_strength_max is not None else 0.0,
            avg_signal_strength=avg_signal_strength,
            final_track_status=self.track_status,
        )


class IFFTrackManager:
    """Maintains active IFF tracks, keyed by aircraft_id.

    Purpose:
        Own exactly the responsibilities Phase 8 Part 2 lists: maintain
        active tracks, assign Track IDs, update existing tracks, delete
        stale tracks, and generate immutable snapshots — extended by
        Phase 8.5 with bounded per-track history and full-lifetime
        completed-track summaries.

    Inputs:
        miss_threshold: consecutive misses before a track is lost
            (configurable; default 5).
        confirmation_threshold: consecutive valid replies before a
            tentative track is confirmed (configurable; default 3).
        enable_logging: Phase 8.5 Part 7 — when True, every produced
            `IFFTrack` snapshot also accumulates in `self.log`, for
            later export via `csv_logging.write_tracks_csv`. Default
            False; `update()`'s return value is identical either way.

    Outputs:
        `update(measurement, ...) -> IFFTrack | None`,
        `get_track(aircraft_id) -> IFFTrack | None`,
        `get_active_tracks() -> list[IFFTrack]`,
        `get_track_history(aircraft_id) -> list[IFFTrack]`,
        `get_completed_track_summaries() -> list[TrackSummary]`.

    Engineering explanation:
        All state lives in a single `dict[aircraft_id, _TrackState]` —
        O(1) lookup, insert, and delete per update; no per-tick scan
        over other tracks is ever needed (Part 4's "association by
        Aircraft_ID" requirement is exactly this dict lookup).
    """

    TENTATIVE_QUALITY = 0.3
    CONFIRMED_QUALITY = 1.0
    MISS_PENALTY = 0.1
    MIN_QUALITY = 0.0

    def __init__(
        self,
        miss_threshold: int = 5,
        confirmation_threshold: int = 3,
        enable_logging: bool = False,
    ) -> None:
        if miss_threshold < 1:
            raise ValueError("miss_threshold must be >= 1")
        if confirmation_threshold < 1:
            raise ValueError("confirmation_threshold must be >= 1")
        self.miss_threshold = miss_threshold
        self.confirmation_threshold = confirmation_threshold
        self.enable_logging = enable_logging
        self.log: list[IFFTrack] = []
        self._tracks: dict[str, _TrackState] = {}
        self._completed_track_summaries: list[TrackSummary] = []
        self._next_track_id = 1

    def update(
        self,
        measurement: DecodedIFFMeasurement,
        relative_velocity: Vector3 | None = None,
        reply_type: ReplyType | None = None,
        signal_strength: float | None = None,
    ) -> IFFTrack | None:
        """Process one decoded measurement for its aircraft_id.

        Inputs:
            measurement: a `DecodedIFFMeasurement` (VALID or NO_REPLY).
            relative_velocity: overrides `measurement.relative_velocity`
                if given (Phase 8 backward compatibility); otherwise the
                measurement's own value is used directly (Phase 8.5
                Part 1 — no second geometry computation needed).
            reply_type: optional context `DecodedIFFMeasurement` does
                not itself carry.
            signal_strength: overrides `measurement.signal_strength` if
                given; otherwise the measurement's own value is used
                (falling back to `NOMINAL_SIGNAL_STRENGTH` only if
                neither is available, e.g. a NO_REPLY measurement).

        Outputs:
            The resulting `IFFTrack` snapshot, or `None` if there was no
            existing track and this measurement was NO_REPLY (nothing
            to initiate a track from).
        """
        aircraft_id = measurement.target_id
        existing = self._tracks.get(aircraft_id)
        is_valid = measurement.reply_status == MeasurementStatus.VALID

        effective_relative_velocity = (
            relative_velocity if relative_velocity is not None else measurement.relative_velocity
        )
        effective_signal_strength = (
            signal_strength
            if signal_strength is not None
            else (measurement.signal_strength if measurement.signal_strength is not None else NOMINAL_SIGNAL_STRENGTH)
        )

        if existing is None:
            if not is_valid:
                return None
            state = self._initiate(
                measurement, effective_relative_velocity, reply_type, effective_signal_strength
            )
            self._tracks[aircraft_id] = state
        else:
            state = existing
            state.accumulate_status_time(measurement.time)
            if is_valid:
                self._apply_valid_reply(
                    state, measurement, effective_relative_velocity, reply_type, effective_signal_strength
                )
            else:
                self._apply_miss(state, measurement)

        snapshot = state.snapshot()
        state.history.append(snapshot)
        if self.enable_logging:
            self.log.append(snapshot)

        if state.track_status == TrackStatus.LOST:
            self._completed_track_summaries.append(state.build_summary())
            del self._tracks[aircraft_id]

        return snapshot

    def get_track(self, aircraft_id: str) -> IFFTrack | None:
        """Return the current snapshot for one aircraft's track, or None."""
        state = self._tracks.get(aircraft_id)
        return state.snapshot() if state else None

    def get_active_tracks(self) -> list[IFFTrack]:
        """Return snapshots of every currently active (non-lost) track."""
        return [state.snapshot() for state in self._tracks.values()]

    def get_track_history(self, aircraft_id: str) -> list[IFFTrack]:
        """Return up to the last `TRACK_HISTORY_MAXLEN` snapshots for one
        active track, oldest first (Phase 8.5 Part 5). Empty list if the
        aircraft has no active track (including a track that has since
        been lost — its history is not retained after removal; see
        `get_completed_track_summaries` for lifetime statistics instead)."""
        state = self._tracks.get(aircraft_id)
        return list(state.history) if state else []

    def get_completed_track_summaries(self) -> list[TrackSummary]:
        """Return a TrackSummary for every track that has been lost so far
        (Phase 8.5 Part 6), in the order they were lost."""
        return list(self._completed_track_summaries)

    def _initiate(
        self,
        measurement: DecodedIFFMeasurement,
        relative_velocity: Vector3 | None,
        reply_type: ReplyType | None,
        signal_strength: float,
    ) -> _TrackState:
        track_id = self._next_track_id
        self._next_track_id += 1
        friend_foe = derive_friend_foe_status(
            measurement.mode, measurement.authentication_result, measurement.reply_status
        )
        state = _TrackState(
            track_id=track_id,
            aircraft_id=measurement.target_id,
            ownship_id=measurement.ownship_id,
            time=measurement.time,
            range_m=measurement.range_m,
            azimuth_deg=measurement.azimuth_deg,
            elevation_deg=measurement.elevation_deg,
            relative_velocity=relative_velocity,
            closing_velocity_mps=measurement.closing_velocity_mps,
            mode=measurement.mode,
            reply_status=measurement.reply_status,
            mode_s_address=measurement.icao_address,
            authentication_result=measurement.authentication_result,
            authentication_status=measurement.authentication_status,
            friend_foe_status=friend_foe,
            track_status=TrackStatus.TENTATIVE,
            track_quality=self.TENTATIVE_QUALITY,
            last_update_time=measurement.time,
            sequence_number=measurement.sequence_number,
            reply_type=reply_type,
            signal_strength=signal_strength,
            propagation_delay=measurement.propagation_delay,
            miss_count=0,
            consecutive_valid_count=1,
            start_time=measurement.time,
            replies_received=1,
        )
        state.record_range_and_signal(measurement.range_m, signal_strength)
        # A track can be confirmed on its very first reply if
        # confirmation_threshold has been configured down to 1 -- this
        # check must run here too, not just in _apply_valid_reply, since
        # _initiate never goes through that method.
        self._apply_confirmation_rule(state)
        return state

    def _apply_valid_reply(
        self,
        state: _TrackState,
        measurement: DecodedIFFMeasurement,
        relative_velocity: Vector3 | None,
        reply_type: ReplyType | None,
        signal_strength: float,
    ) -> None:
        state.time = measurement.time
        state.range_m = measurement.range_m
        state.azimuth_deg = measurement.azimuth_deg
        state.elevation_deg = measurement.elevation_deg
        state.relative_velocity = relative_velocity
        state.closing_velocity_mps = measurement.closing_velocity_mps
        state.mode = measurement.mode
        state.reply_status = measurement.reply_status
        state.mode_s_address = measurement.icao_address
        state.authentication_result = measurement.authentication_result
        state.authentication_status = measurement.authentication_status
        state.friend_foe_status = derive_friend_foe_status(
            measurement.mode, measurement.authentication_result, measurement.reply_status
        )
        state.signal_strength = signal_strength
        state.propagation_delay = measurement.propagation_delay
        state.last_update_time = measurement.time
        state.sequence_number = measurement.sequence_number
        state.reply_type = reply_type

        state.miss_count = 0
        state.consecutive_valid_count += 1
        state.replies_received += 1
        state.record_range_and_signal(measurement.range_m, signal_strength)

        self._apply_confirmation_rule(state)

    def _apply_confirmation_rule(self, state: _TrackState) -> None:
        """Shared by `_initiate` and `_apply_valid_reply`: set quality for
        the current status, confirming once `consecutive_valid_count`
        reaches `confirmation_threshold`."""
        if state.track_status == TrackStatus.TENTATIVE:
            state.track_quality = self.TENTATIVE_QUALITY
            if state.consecutive_valid_count >= self.confirmation_threshold:
                state.track_status = TrackStatus.CONFIRMED
                state.track_quality = self.CONFIRMED_QUALITY
        elif state.track_status == TrackStatus.CONFIRMED:
            state.track_quality = self.CONFIRMED_QUALITY

    def _apply_miss(self, state: _TrackState, measurement: DecodedIFFMeasurement) -> None:
        # Range/azimuth/elevation/mode/mode_s_address/authentication/signal
        # strength/propagation delay/relative & closing velocity are
        # deliberately left untouched: a NO_REPLY carries no new geometry
        # or identity data to update them from (see `IFFTrack.range_m`'s
        # docstring).
        state.time = measurement.time
        state.reply_status = measurement.reply_status
        state.friend_foe_status = derive_friend_foe_status(
            state.mode, state.authentication_result, measurement.reply_status
        )
        state.sequence_number = measurement.sequence_number

        state.miss_count += 1
        state.replies_missed += 1
        state.consecutive_valid_count = 0
        state.track_quality = max(self.MIN_QUALITY, state.track_quality - self.MISS_PENALTY)

        if state.miss_count >= self.miss_threshold:
            state.track_status = TrackStatus.LOST
