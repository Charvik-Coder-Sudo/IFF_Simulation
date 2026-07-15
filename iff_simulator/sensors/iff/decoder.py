"""Decodes a matched reply (or a timeout) into a DecodedIFFMeasurement.

Purpose:
    Implements `ModeDecoder`, which unpacks a `MatchResult`'s payload
    into a `DecodedIFFMeasurement` — or, if no reply matched, produces
    the `NO_REPLY` measurement. Never estimates a value: every field is
    either copied verbatim from the interrogation/reply/payload, or a
    fixed `None`/`"UNKNOWN"` default when that mode has no such concept.

Inputs:
    A `MatchResult`.

Outputs:
    A `DecodedIFFMeasurement`.

Engineering explanation:
    Geometry (range/azimuth/elevation) is copied verbatim from the
    `InterrogationMessage` — itself sourced from `GeometryEngine` via
    Phase 4's `TargetSelector` — never recomputed here, satisfying this
    phase's "reuse GeometryEngine, no duplicated math" requirement.
"""

from __future__ import annotations

from .authentication import AuthenticationResult, classify_friendly_status, derive_authentication_status
from .matcher import MatchResult
from .measurement import DecodedIFFMeasurement, MeasurementStatus
from .mode5 import Mode5Level1Payload, Mode5Level2Payload
from .mode_s import ModeSPayload


class ModeDecoder:
    """Unpacks a matched reply's logical payload into a DecodedIFFMeasurement.

    Purpose:
        The single place a `ReplyMessage`'s mode-specific payload is
        translated into the cross-mode `DecodedIFFMeasurement` schema.

    Inputs:
        `decode(match_result)`.

    Outputs:
        A `DecodedIFFMeasurement`.

    Engineering explanation:
        Stateless and deterministic: the same `MatchResult` always
        decodes to the same `DecodedIFFMeasurement`. Dispatches on the
        payload's concrete type (`ModeSPayload` / `Mode5Level1Payload` /
        `Mode5Level2Payload`) rather than `match_result.interrogation.mode`,
        since the payload is the actual source of truth for what was
        received.

        Phase 8.5 Part 7: `enable_logging=True` makes every decoded
        measurement also accumulate in `self.log`, for later export via
        `csv_logging.write_decoded_csv`. Default is `False`, and with it
        `decode()`'s return value and every existing test's behavior are
        completely unchanged — purely additive.
    """

    def __init__(self, enable_logging: bool = False) -> None:
        self.enable_logging = enable_logging
        self.log: list[DecodedIFFMeasurement] = []

    def decode(self, match_result: MatchResult) -> DecodedIFFMeasurement:
        """Decode one MatchResult into a DecodedIFFMeasurement.

        Inputs:
            match_result: the `MatchResult` to decode (matched or timed out).
        Outputs:
            A `DecodedIFFMeasurement` with `reply_status` set to
            `MeasurementStatus.VALID` (matched) or
            `MeasurementStatus.NO_REPLY` (not matched/timed out).
        """
        measurement = self._decode(match_result)
        if self.enable_logging:
            self.log.append(measurement)
        return measurement

    def _decode(self, match_result: MatchResult) -> DecodedIFFMeasurement:
        interrogation = match_result.interrogation

        if match_result.propagated_reply is None:
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
                reply_status=MeasurementStatus.NO_REPLY,
                processing_delay=None,
                propagation_delay=None,
                arrival_time=None,
                sequence_number=interrogation.sequence_number,
                closing_velocity_mps=interrogation.closing_velocity_mps,
                relative_velocity=interrogation.relative_velocity,
                authentication_status=AuthenticationResult.NOT_APPLICABLE,
            )

        propagated = match_result.propagated_reply
        reply = propagated.reply
        payload = reply.payload

        if isinstance(payload, ModeSPayload):
            icao_address = payload.icao_address
            identity = payload.identity
            mission = None
        elif isinstance(payload, Mode5Level1Payload):
            icao_address = None
            identity = payload.friendly_status
            mission = payload.mission_code
        elif isinstance(payload, Mode5Level2Payload):
            # Mode 5 L2's own payload carries no friendly/identity field
            # (only Platform Address, Mission, Time, Additional Status);
            # classify purely from the reply's own authentication result,
            # never inventing an identity value that wasn't decoded.
            icao_address = None
            identity = classify_friendly_status("UNKNOWN", reply.authenticated)
            mission = payload.mission
        else:  # pragma: no cover - defensive; every generator uses a known payload type
            raise TypeError(f"Unknown reply payload type: {type(payload).__name__}")

        return DecodedIFFMeasurement(
            measurement_id=interrogation.sequence_number,
            time=interrogation.time,
            target_id=interrogation.target_id,
            ownship_id=interrogation.ownship_id,
            mode=interrogation.mode,
            range_m=interrogation.range_m,
            azimuth_deg=interrogation.azimuth_deg,
            elevation_deg=interrogation.elevation_deg,
            icao_address=icao_address,
            authentication_result=reply.authenticated,
            identity=identity,
            mission=mission,
            reply_status=MeasurementStatus.VALID,
            processing_delay=reply.processing_delay,
            propagation_delay=propagated.propagation_delay_us,
            arrival_time=propagated.arrival_time,
            sequence_number=interrogation.sequence_number,
            closing_velocity_mps=interrogation.closing_velocity_mps,
            relative_velocity=interrogation.relative_velocity,
            authentication_status=derive_authentication_status(interrogation.mode, reply.authenticated),
            signal_strength=propagated.signal_strength,
        )
