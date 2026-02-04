"""
CSV data logging for simulation runs: time, temperatures, power, depth, and optional state.
"""

import csv
import os
from typing import Optional
import numpy as np


def open_log(
    filepath: str,
    include_state: bool = True,
) -> "CSVLogger":
    """
    Create a CSV logger and write header.
    filepath: path to CSV file (directory will be created if needed).
    include_state: if True, add a 'state' column.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)) or ".", exist_ok=True)
    f = open(filepath, "w", newline="")
    writer = csv.writer(f)
    header = [
        "t_s",
        "T_X_C", "T_Y_C", "T_Z_C",
        "P_X_W", "P_Y_W", "P_Z_W",
        "depth_X_cm", "depth_Y_cm", "depth_Z_cm",
    ]
    if include_state:
        header.append("state")
    writer.writerow(header)
    return CSVLogger(f, writer, include_state)


class CSVLogger:
    """Append rows for each simulation step."""

    def __init__(self, file_handle, writer: csv.writer, include_state: bool):
        self._f = file_handle
        self._writer = writer
        self._include_state = include_state

    def log_row(
        self,
        t: float,
        T: np.ndarray,
        P: np.ndarray,
        depth: np.ndarray,
        state: Optional[str] = None,
    ) -> None:
        """Write one row. T, P, depth are 1D arrays of length 3 (X, Y, Z)."""
        row = [t, float(T[0]), float(T[1]), float(T[2]),
               float(P[0]), float(P[1]), float(P[2]),
               float(depth[0]), float(depth[1]), float(depth[2])]
        if self._include_state and state is not None:
            row.append(state)
        elif self._include_state:
            row.append("")
        self._writer.writerow(row)
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "CSVLogger":
        return self

    def __exit__(self, *args) -> None:
        self.close()
