"""
Finite-state machine for the coil controller: OFF, ARMED, FIRING, FAULT.
"""

from enum import Enum
from typing import Optional


class ControllerState(Enum):
    OFF = "OFF"
    ARMED = "ARMED"
    FIRING = "FIRING"
    FAULT = "FAULT"


# Valid transitions: (from_state, to_state)
TRANSITIONS = {
    (ControllerState.OFF, ControllerState.ARMED),
    (ControllerState.ARMED, ControllerState.OFF),
    (ControllerState.ARMED, ControllerState.FIRING),
    (ControllerState.FIRING, ControllerState.OFF),
    (ControllerState.FIRING, ControllerState.FAULT),
    (ControllerState.FAULT, ControllerState.OFF),
}


def can_transition(from_state: ControllerState, to_state: ControllerState) -> bool:
    """Return True if transition from_state -> to_state is allowed."""
    return (from_state, to_state) in TRANSITIONS


def transition(
    current: ControllerState,
    to_state: ControllerState,
    fault_reason: Optional[str] = None,
) -> tuple[ControllerState, Optional[str]]:
    """
    Attempt transition to to_state. Returns (new_state, fault_reason).
    fault_reason is set when entering FAULT and cleared when leaving.
    If transition is invalid, returns (current, fault_reason) unchanged.
    """
    if not can_transition(current, to_state):
        return current, fault_reason
    if to_state == ControllerState.FAULT and fault_reason is None:
        fault_reason = "unknown"
    if to_state != ControllerState.FAULT:
        fault_reason = None
    return to_state, fault_reason
