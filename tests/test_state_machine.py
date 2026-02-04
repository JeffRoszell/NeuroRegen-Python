"""Tests for the controller finite-state machine."""

import pytest
from src.neuroregen.state_machine import (
    ControllerState,
    can_transition,
    transition,
    TRANSITIONS,
)


def test_transitions_defined():
    assert (ControllerState.OFF, ControllerState.ARMED) in TRANSITIONS
    assert (ControllerState.ARMED, ControllerState.FIRING) in TRANSITIONS
    assert (ControllerState.FIRING, ControllerState.FAULT) in TRANSITIONS
    assert (ControllerState.FAULT, ControllerState.OFF) in TRANSITIONS


def test_off_to_armed():
    assert can_transition(ControllerState.OFF, ControllerState.ARMED)
    s, r = transition(ControllerState.OFF, ControllerState.ARMED)
    assert s == ControllerState.ARMED
    assert r is None


def test_armed_to_firing():
    assert can_transition(ControllerState.ARMED, ControllerState.FIRING)
    s, r = transition(ControllerState.ARMED, ControllerState.FIRING)
    assert s == ControllerState.FIRING
    assert r is None


def test_firing_to_fault():
    assert can_transition(ControllerState.FIRING, ControllerState.FAULT)
    s, r = transition(ControllerState.FIRING, ControllerState.FAULT, fault_reason="over_temperature")
    assert s == ControllerState.FAULT
    assert r == "over_temperature"


def test_fault_to_off():
    assert can_transition(ControllerState.FAULT, ControllerState.OFF)
    s, r = transition(ControllerState.FAULT, ControllerState.OFF, fault_reason="over_temperature")
    assert s == ControllerState.OFF
    assert r is None


def test_invalid_transitions():
    assert not can_transition(ControllerState.OFF, ControllerState.FIRING)
    assert not can_transition(ControllerState.ARMED, ControllerState.FAULT)
    assert not can_transition(ControllerState.FAULT, ControllerState.ARMED)
    s, _ = transition(ControllerState.OFF, ControllerState.FIRING)
    assert s == ControllerState.OFF
