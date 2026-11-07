"""Exact-phase adapter regressions for the released core 0.3.1 stack."""

import math

import numpy as np
import pennylane as qml
import pytest

from rqm_pennylane.gates import apply_quaternion_rotation


@pytest.mark.parametrize("axis,gate", [(0, qml.RX), (1, qml.RY), (2, qml.RZ)])
@pytest.mark.parametrize("angle", [2 * math.pi, -2 * math.pi, 4 * math.pi, 1e-10, -0.73])
def test_quaternion_rotation_retains_exact_phase_under_control(axis, gate, angle):
    """Use framework rotations, not core's conversion, as the independent oracle."""
    components = [math.cos(angle / 2), 0.0, 0.0, 0.0]
    components[axis + 1] = math.sin(angle / 2)
    actual = apply_quaternion_rotation(components, wires=1)
    expected = gate(angle, wires=1)
    np.testing.assert_allclose(qml.matrix(actual), qml.matrix(expected), atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        qml.matrix(qml.ctrl(actual, control=0), wire_order=[0, 1]),
        qml.matrix(qml.ctrl(expected, control=0), wire_order=[0, 1]),
        atol=1e-12,
        rtol=0,
    )


def test_negative_quaternion_is_not_sign_canonicalized():
    positive = qml.matrix(apply_quaternion_rotation([0.5, 0.5, 0.5, 0.5], wires=0))
    negative = qml.matrix(apply_quaternion_rotation([-0.5, -0.5, -0.5, -0.5], wires=0))
    np.testing.assert_allclose(negative, -positive, atol=1e-12, rtol=0)
