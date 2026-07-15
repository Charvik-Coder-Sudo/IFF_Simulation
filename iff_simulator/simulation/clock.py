"""A simple discrete-time simulation clock.

Purpose:
    Implements `SimulationClock`, the single source of truth for "what
    time is it" in a running simulation, independent of any particular
    aircraft, sensor, or scenario.

Inputs:
    `start_time`, `dt` (timestep), and `end_time` at construction.

Outputs:
    A monotonically-advancing clock: `step()` advances it, `reset()`
    rewinds it, `finished()` reports whether it has reached the end.

Engineering explanation:
    This clock does no simulation work itself — no motion propagation,
    no geometry, no IFF logic. It only tracks elapsed simulated time, so
    every future module that needs to know "what time is it" (Ownship,
    Geometry, PSR, IFF, Scheduler) can share one clock instance instead
    of tracking time independently and drifting out of sync.
"""

from __future__ import annotations


class SimulationClock:
    """Tracks simulated time advancing in fixed steps from start to end.

    Purpose:
        Provide a single, reusable notion of simulated time that other
        components can step, reset, and query.

    Inputs:
        start_time: the time value the clock begins at.
        dt: the fixed timestep applied on each `step()` call.
        end_time: the time value at which the clock is considered
            finished.

    Outputs:
        `current_time()`, `step()`, `reset()`, `finished()`.

    Engineering explanation:
        `step()` clamps to `end_time` so the clock never overshoots the
        configured simulation horizon, which keeps any future
        time-indexed lookups (e.g. `World.step()`) safely within the
        range of recorded/simulated data.
    """

    def __init__(self, start_time: float = 0.0, dt: float = 1.0, end_time: float = 0.0) -> None:
        self.dt = dt
        self.end_time = end_time
        self._start_time = start_time
        self._current_time = start_time

    def current_time(self) -> float:
        """Return the clock's current simulated time."""
        return self._current_time

    def step(self) -> float:
        """Advance the clock by `dt`, clamped to `end_time`.

        Returns:
            The new current time.
        """
        self._current_time = min(self._current_time + self.dt, self.end_time)
        return self._current_time

    def reset(self) -> None:
        """Reset the clock back to its configured start time."""
        self._current_time = self._start_time

    def finished(self) -> bool:
        """Return whether the clock has reached its configured end time."""
        return self._current_time >= self.end_time
