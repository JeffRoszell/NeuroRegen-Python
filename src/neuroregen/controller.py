"""
Controller: FSM state, axis enable, run loop with stepwise sim, CSV log, and live plot.
"""

import os
import numpy as np
from typing import Callable, Optional

from .state_machine import ControllerState, transition
from .simulation import run_simulation_stepwise
from .coil import f_to_c
from .csv_logger import open_log


class Controller:
    """
    Holds FSM state, config, axis enable flags, and runs the firing loop
    with optional CSV logging and live plotting.
    """

    def __init__(self, config: dict, csv_dir: str = "outputs/logs", log_run: bool = True):
        self.config = config
        self.state = ControllerState.OFF
        self.fault_reason: Optional[str] = None
        self.axis_enabled = [True, True, True]
        self.stop_requested = False
        self.log_run = log_run
        self.csv_dir = csv_dir
        self.csv_path: Optional[str] = None
        self._limit_c = f_to_c(config["temp_limit_f"])
        self._live_plot_update: Optional[Callable] = None
        self._live_plot_every_n = 5  # update plot every N steps to reduce lag

    def arm(self) -> bool:
        """Transition OFF -> ARMED. Returns True if successful."""
        new_state, _ = transition(self.state, ControllerState.ARMED, self.fault_reason)
        if new_state != self.state:
            self.state = new_state
            return True
        return False

    def disarm(self) -> bool:
        """Transition ARMED -> OFF. Returns True if successful."""
        new_state, _ = transition(self.state, ControllerState.OFF, self.fault_reason)
        if new_state != self.state:
            self.state = new_state
            return True
        return False

    def start(self) -> bool:
        """Transition ARMED -> FIRING. Returns True if successful."""
        new_state, _ = transition(self.state, ControllerState.FIRING, self.fault_reason)
        if new_state != self.state:
            self.state = new_state
            return True
        return False

    def stop(self) -> bool:
        """Transition FIRING -> OFF (user stop). Returns True if successful."""
        new_state, _ = transition(self.state, ControllerState.OFF, self.fault_reason)
        if new_state != self.state:
            self.state = new_state
            self.stop_requested = True
            return True
        return False

    def clear_fault(self) -> bool:
        """Transition FAULT -> OFF. Returns True if successful."""
        new_state, reason = transition(self.state, ControllerState.OFF, self.fault_reason)
        if new_state != self.state:
            self.state = new_state
            self.fault_reason = reason
            return True
        return False

    def set_fault(self, reason: str) -> None:
        """Set state to FAULT (from FIRING)."""
        new_state, reason = transition(self.state, ControllerState.FAULT, reason)
        if new_state == ControllerState.FAULT:
            self.state = new_state
            self.fault_reason = reason

    def set_live_plot_update(self, update_fn: Optional[Callable]) -> None:
        """Set the live plot update function (t, T, P, depth) -> None."""
        self._live_plot_update = update_fn

    def run_firing_loop(self) -> tuple:
        """
        Run the stepwise simulation while state is FIRING.
        Stops on stop_requested, over-temp (FAULT), or end of sim_time.
        Returns (t, T, P, depth) arrays from the run.
        """
        self.stop_requested = False
        csv_handle = None
        t_list, T_list, P_list, depth_list = [], [], [], []

        if self.log_run and self.csv_dir:
            os.makedirs(self.csv_dir, exist_ok=True)
            from datetime import datetime

            base = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.csv_path = os.path.join(self.csv_dir, f"run_{base}.csv")
            csv_handle = open_log(self.csv_path, include_state=True)

        try:
            gen = run_simulation_stepwise(
                config=self.config,
                axis_enabled=self.axis_enabled,
                stop_check=lambda: self.stop_requested,
            )
            step_count = 0
            for payload in gen:
                if payload[0] == "final":
                    _, t_arr, T_arr, P_arr, d_arr = payload
                    t_list, T_list = list(t_arr), list(T_arr)
                    P_list, depth_list = list(P_arr), list(d_arr)
                    break
                _, k, t_k, T_k, P_k, depth_k = payload
                t_list.append(t_k)
                T_list.append(T_k)
                P_list.append(P_k)
                depth_list.append(depth_k)
                if csv_handle:
                    csv_handle.log_row(t_k, T_k, P_k, depth_k, self.state.value)

                if any(T_k >= self._limit_c):
                    self.set_fault("over_temperature")
                    break

                if self._live_plot_update and (step_count % self._live_plot_every_n == 0):
                    self._live_plot_update(t_k, T_k, P_k, depth_k)
                step_count += 1
        finally:
            if csv_handle:
                csv_handle.close()

        if self.state == ControllerState.FIRING:
            self.state = ControllerState.OFF

        if not t_list:
            return np.array([]), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3))
        return (
            np.array(t_list),
            np.array(T_list).reshape(-1, 3),
            np.array(P_list).reshape(-1, 3),
            np.array(depth_list).reshape(-1, 3),
        )
