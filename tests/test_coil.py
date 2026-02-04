"""
Unit tests for coil geometry, resistance, B-field, and depth.
"""

import pytest
import numpy as np

# Add project root so we can import src
import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.neuroregen.coil import (
    Axis,
    coil_geom,
    resistance,
    B_loop,
    effective_depth_cm,
    c_to_f,
    f_to_c,
)


def test_c_to_f():
    assert c_to_f(0) == 32
    assert c_to_f(100) == 212


def test_f_to_c():
    assert f_to_c(32) == 0
    assert f_to_c(212) == 100


def test_coil_geom():
    a = Axis("X", 1.2, 80, 20, 45)
    R, L, A, S, m = coil_geom(a)
    assert R == 0.04
    assert L > 0 and A > 0 and S > 0 and m > 0


def test_resistance_positive():
    R = resistance(1.0, 1e-6, 20)
    assert R > 0


def test_B_loop_shape():
    z = np.linspace(0, 0.03, 100)
    B = B_loop(1.0, 0.04, z, 20)
    assert B.shape == z.shape
