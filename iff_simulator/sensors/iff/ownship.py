"""The airborne platform carrying the IFF interrogator.

Purpose:
    Implements `Ownship`, the platform whose position/velocity/heading
    define "where the IFF interrogator is" at any point in the
    simulation. This introduces the runtime world model only — no
    interrogation, reply, or geometry logic lives here.

Inputs:
    An `aircraft_id` designating which `Scenario` aircraft is "self",
    and a reference to that `Scenario`.

Outputs:
    `position`, `velocity`, and `heading` properties that always reflect
    the designated aircraft's current recorded state, plus platform/
    sensor configuration fields (pitch, roll, maximum range, beam width,
    beam height, interrogation rate, operating modes).

Engineering explanation:
    `position`/`velocity`/`heading` are deliberately *properties*, not
    stored fields copied at construction time: they read straight
    through to `Scenario.get_state(aircraft_id)` on every access. This
    guarantees Ownship can never hold a stale or duplicated copy of its
    own kinematics — as soon as `World.step()` advances the Scenario's
    current-state cursor, Ownship's position/velocity/heading reflect it
    automatically, with nothing to keep in sync by hand. Pitch and roll
    have no equivalent field in the recorded ground truth (which is a
    2D dataset), so they are genuine stored configuration values,
    defaulting to 0.0.
"""

from __future__ import annotations

from ...domain import Scenario, Vector3


class Ownship:
    """The aircraft designated as the IFF interrogator platform.

    Purpose:
        Represent "self" in the simulation: the platform whose sensor
        (maximum range, beam width, interrogation rate, operating
        modes) will, in a later phase, interrogate other aircraft. This
        phase only wires up the platform's identity and live kinematic
        state.

    Inputs:
        aircraft_id: the `Scenario` aircraft designated as Ownship
            (Phase 2 designates Target 1).
        scenario: the `Scenario` this Ownship's kinematics are read
            from.
        pitch, roll: platform attitude, degrees; not present in the
            recorded 2D ground truth, so they default to 0.0 and are
            not computed or propagated in this phase.
        maximum_range: interrogator maximum range, meters; a
            configuration value, not a computed geometric range.
        beam_width: interrogator antenna beam width in azimuth, degrees
            (full angle, e.g. 60 means +/-30 degrees off boresight).
        beam_height: interrogator antenna beam width in elevation,
            degrees (full angle, same convention as beam_width). Added
            for the Phase 4 antenna-beam selection check; defaults to
            0.0 so every existing caller that does not pass it keeps
            its prior behavior unchanged.
        interrogation_rate: interrogations per second the sensor is
            configured for.
        operating_modes: IFF modes the interrogator is configured to
            use; empty by default since no IFF logic exists yet.

    Outputs:
        `position`, `velocity`, `heading` (live, read-through
        properties), plus the stored platform/sensor configuration
        fields listed above.

    Engineering explanation:
        No interrogation, reply, or geometry computation happens here —
        this class only answers "where and what is the interrogating
        platform," which is the prerequisite every later IFF phase
        needs before it can compute anything geometric.
    """

    def __init__(
        self,
        aircraft_id: str,
        scenario: Scenario,
        pitch: float = 0.0,
        roll: float = 0.0,
        maximum_range: float = 0.0,
        beam_width: float = 0.0,
        beam_height: float = 0.0,
        interrogation_rate: float = 0.0,
        operating_modes: list[str] | None = None,
    ) -> None:
        self.aircraft_id = aircraft_id
        self._scenario = scenario
        self.pitch = pitch
        self.roll = roll
        self.maximum_range = maximum_range
        self.beam_width = beam_width
        self.beam_height = beam_height
        self.interrogation_rate = interrogation_rate
        self.operating_modes: list[str] = list(operating_modes) if operating_modes else []

    @property
    def position(self) -> Vector3:
        """The Ownship's current position, read live from the Scenario."""
        return self._scenario.get_state(self.aircraft_id).position

    @property
    def velocity(self) -> Vector3:
        """The Ownship's current velocity, read live from the Scenario."""
        return self._scenario.get_state(self.aircraft_id).velocity

    @property
    def heading(self) -> float:
        """The Ownship's current heading, read live from the Scenario."""
        return self._scenario.get_state(self.aircraft_id).heading
