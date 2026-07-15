"""IFF interrogation mode selection: which protocol family to request.

Purpose:
    Defines `IFFMode` (the three in-scope modes) and the `ModeSelectionPolicy`
    strategy that decides which mode to request for a given target.

Inputs:
    A `SelectedTarget` (Phase 4 output).

Outputs:
    An `IFFMode` value.

Scope:
    Mode S, Mode 5 Level 1, and Mode 5 Level 2 only. Legacy modes
    (1, 2, 3/A, C, 4) are explicitly out of scope for this simulator and
    are not represented here at all — adding them later means adding
    new `IFFMode` members and a new policy, never modifying this one.

Engineering explanation:
    Mode selection is a Strategy (like Phase 4's `SelectionPolicy`) so
    `InterrogationScheduler` never hard-codes "every target gets Mode
    S" — it asks an injected `ModeSelectionPolicy`. Only the default,
    trivial policy is implemented this phase; identity-based mode
    selection (friendly -> Mode 5 L2, unknown -> Mode 5 L1, civil ->
    Mode S) is an explicitly deferred extension point.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, unique

from .selected_target import SelectedTarget


@unique
class IFFMode(Enum):
    """The IFF interrogation modes this simulator supports.

    Purpose:
        Enumerate exactly the in-scope modes for this phase.

    Engineering explanation:
        Deliberately excludes Mode 1/2/3A/C/4 — those are legacy
        extensions for a later phase, not represented as enum members
        here at all (not even as unused/reserved values), so this type
        cannot accidentally be used to request them.
    """

    MODE_S = "MODE_S"
    MODE5_L1 = "MODE5_L1"
    MODE5_L2 = "MODE5_L2"


class ModeSelectionPolicy(ABC):
    """Abstract strategy: which IFFMode to request for a given target.

    Purpose:
        Let `InterrogationScheduler` depend on an interface rather than
        a fixed rule, so later mode-selection logic (based on IFF
        identity, mission rules, etc.) can be injected without
        modifying the scheduler.

    Inputs:
        selected_target: the `SelectedTarget` about to be interrogated.

    Outputs:
        An `IFFMode`.
    """

    @abstractmethod
    def select_mode(self, selected_target: SelectedTarget) -> IFFMode:
        """Decide which IFFMode to request for this target.

        Purpose:
            The single decision point every mode-selection rule funnels
            through.
        Inputs:
            selected_target: the target about to be interrogated.
        Outputs:
            An `IFFMode`.
        Engineering reasoning:
            Takes only a `SelectedTarget` (geometry/identity already
            resolved by earlier phases) — never recomputes anything
            about the target itself.
        """
        raise NotImplementedError


class DefaultModeSelectionPolicy(ModeSelectionPolicy):
    """Default mode selection: every target gets Mode S.

    Purpose:
        The trivial, always-correct default: request Mode S for every
        target, regardless of who it is.

    Engineering explanation:
        A future policy (e.g. "friendly -> Mode5 L2, unknown -> Mode5
        L1, civil -> Mode S") is a documented extension point, not
        implemented here — this phase only wires up the injection
        point (`ModeSelectionPolicy`), per the "future extension hooks
        only" instruction.
    """

    def select_mode(self, selected_target: SelectedTarget) -> IFFMode:
        return IFFMode.MODE_S
