"""
tests/test_gates.py — Tests for rqm_pennylane.gates.
"""

import math

import numpy as np
import pennylane as qml
import pytest

from rqm_pennylane.gates import (
    RQMRotation,
    apply_quaternion_rotation,
    parameterized_su2,
)


@pytest.fixture
def dev():
    return qml.device("default.qubit", wires=1)


class TestRQMRotation:
    def test_returns_scalar_expectation(self, dev):
        @qml.qnode(dev)
        def circuit(phi, theta, omega):
            RQMRotation(phi, theta, omega, wires=0)
            return qml.expval(qml.PauliZ(0))

        result = circuit(0.1, 0.2, 0.3)
        assert math.isfinite(float(result))

    def test_identity_rotation_expectation_is_one(self, dev):
        @qml.qnode(dev)
        def circuit():
            RQMRotation(0.0, 0.0, 0.0, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert abs(float(circuit()) - 1.0) < 1e-10

    def test_pi_rotation_about_y_flips_state(self, dev):
        @qml.qnode(dev)
        def circuit():
            # qml.Rot(phi=0, theta=pi, omega=0) → RY(pi) = X up to phase
            RQMRotation(0.0, math.pi, 0.0, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert abs(float(circuit()) - (-1.0)) < 1e-10

    def test_differentiable(self, dev):
        @qml.qnode(dev)
        def circuit(params):
            RQMRotation(params[0], params[1], params[2], wires=0)
            return qml.expval(qml.PauliZ(0))

        params = qml.numpy.array([0.1, 0.2, 0.3], requires_grad=True)
        grad_fn = qml.grad(circuit)
        grads = grad_fn(params)
        assert grads.shape == (3,)
        assert all(math.isfinite(float(g)) for g in grads)


class TestApplyQuaternionRotation:
    def test_identity_quaternion_no_rotation(self, dev):
        @qml.qnode(dev)
        def circuit():
            apply_quaternion_rotation([1, 0, 0, 0], wires=0)
            return qml.expval(qml.PauliZ(0))

        assert abs(float(circuit()) - 1.0) < 1e-10

    def test_returns_finite_expectation(self, dev):
        @qml.qnode(dev)
        def circuit():
            apply_quaternion_rotation([0.5, 0.5, 0.5, 0.5], wires=0)
            return qml.expval(qml.PauliZ(0))

        result = float(circuit())
        assert math.isfinite(result)

    def test_180_x_rotation_flips_z_expectation(self, dev):
        # Quaternion for 180° around X: (0, 1, 0, 0) → RX(pi) → flips Z
        @qml.qnode(dev)
        def circuit():
            apply_quaternion_rotation([0, 1, 0, 0], wires=0)
            return qml.expval(qml.PauliZ(0))

        assert abs(float(circuit()) - (-1.0)) < 1e-10


class TestParameterizedSu2:
    def test_identity_angles_give_z_expectation_one(self, dev):
        @qml.qnode(dev)
        def circuit():
            parameterized_su2(0.0, 0.0, 0.0, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert abs(float(circuit()) - 1.0) < 1e-10

    def test_ry_pi_flips_z_expectation(self, dev):
        @qml.qnode(dev)
        def circuit():
            # alpha=0, beta=pi, gamma=0 → pure Ry(pi)
            parameterized_su2(0.0, math.pi, 0.0, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert abs(float(circuit()) - (-1.0)) < 1e-10

    def test_differentiable(self, dev):
        @qml.qnode(dev)
        def circuit(params):
            parameterized_su2(params[0], params[1], params[2], wires=0)
            return qml.expval(qml.PauliZ(0))

        params = qml.numpy.array([0.1, 0.2, 0.3], requires_grad=True)
        grads = qml.grad(circuit)(params)
        assert grads.shape == (3,)
        assert all(math.isfinite(float(g)) for g in grads)
