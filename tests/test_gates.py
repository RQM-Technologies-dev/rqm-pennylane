"""
tests/test_gates.py — Tests for rqm_pennylane.gates.
"""

import math

import numpy as np
import pennylane as qml
import pytest

from rqm_pennylane.gates import (
    RQMRotation,
    accumulate_gate_quaternions,
    apply_quaternion_rotation,
    canonicalize_gate_quaternion,
    gate_to_quaternion,
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


class TestGateToQuaternion:
    """Tests for gate_to_quaternion (theory §§9, 21)."""

    def test_identity_gate(self):
        q = gate_to_quaternion("I")
        assert abs(q.w - 1.0) < 1e-12
        assert abs(q.x) < 1e-12
        assert abs(q.y) < 1e-12
        assert abs(q.z) < 1e-12

    def test_x_gate_is_pi_rotation_about_x(self):
        # X = Rx(π) → q ≈ (0, 1, 0, 0)
        q = gate_to_quaternion("X")
        assert abs(q.x - 1.0) < 1e-9
        assert abs(q.y) < 1e-9
        assert abs(q.z) < 1e-9

    def test_y_gate_is_pi_rotation_about_y(self):
        q = gate_to_quaternion("Y")
        assert abs(q.y - 1.0) < 1e-9
        assert abs(q.x) < 1e-9
        assert abs(q.z) < 1e-9

    def test_z_gate_is_pi_rotation_about_z(self):
        q = gate_to_quaternion("Z")
        assert abs(q.z - 1.0) < 1e-9
        assert abs(q.x) < 1e-9
        assert abs(q.y) < 1e-9

    def test_rx_quaternion_formula(self):
        # Rx(θ) ↔ cos(θ/2) + i·sin(θ/2)
        theta = 1.2
        q = gate_to_quaternion("Rx", angle=theta)
        assert abs(q.w - math.cos(theta / 2)) < 1e-12
        assert abs(q.x - math.sin(theta / 2)) < 1e-12
        assert abs(q.y) < 1e-12
        assert abs(q.z) < 1e-12

    def test_ry_quaternion_formula(self):
        theta = 0.8
        q = gate_to_quaternion("Ry", angle=theta)
        assert abs(q.w - math.cos(theta / 2)) < 1e-12
        assert abs(q.y - math.sin(theta / 2)) < 1e-12
        assert abs(q.x) < 1e-12
        assert abs(q.z) < 1e-12

    def test_rz_quaternion_formula(self):
        theta = 2.1
        q = gate_to_quaternion("Rz", angle=theta)
        assert abs(q.w - math.cos(theta / 2)) < 1e-12
        assert abs(q.z - math.sin(theta / 2)) < 1e-12
        assert abs(q.x) < 1e-12
        assert abs(q.y) < 1e-12

    def test_h_gate_has_equal_x_and_z(self):
        # H ↔ π-rotation about (x+z)/√2 → q imaginary part ∝ (i + k)
        q = gate_to_quaternion("H")
        assert abs(abs(q.x) - abs(q.z)) < 1e-12
        assert abs(q.y) < 1e-12

    def test_result_is_unit_quaternion(self):
        for name, angle in [("I", None), ("X", None), ("Y", None), ("Z", None),
                             ("H", None), ("S", None), ("T", None),
                             ("Rx", 1.0), ("Ry", 1.0), ("Rz", 1.0)]:
            q = gate_to_quaternion(name, angle=angle)
            norm = math.sqrt(q.w ** 2 + q.x ** 2 + q.y ** 2 + q.z ** 2)
            assert abs(norm - 1.0) < 1e-12, f"Gate {name} quaternion not unit: {norm}"

    def test_parameterized_gate_without_angle_raises(self):
        import pytest
        with pytest.raises(ValueError, match="parameterised"):
            gate_to_quaternion("Rx")

    def test_unknown_gate_raises(self):
        import pytest
        with pytest.raises(ValueError, match="not a recognised"):
            gate_to_quaternion("CCNOT")

    def test_case_insensitive(self):
        q_lower = gate_to_quaternion("x")
        q_upper = gate_to_quaternion("X")
        assert abs(q_lower.w - q_upper.w) < 1e-12
        assert abs(q_lower.x - q_upper.x) < 1e-12


class TestAccumulateGateQuaternions:
    """Tests for accumulate_gate_quaternions (theory §§10, 11)."""

    def test_single_gate_returns_that_gate(self):
        q_x = gate_to_quaternion("X")
        result = accumulate_gate_quaternions([q_x])
        norm = math.sqrt(result.w ** 2 + result.x ** 2 + result.y ** 2 + result.z ** 2)
        assert abs(norm - 1.0) < 1e-12

    def test_x_x_is_identity(self):
        # X·X = I in SU(2) (up to global sign)
        q_x = gate_to_quaternion("X")
        result = accumulate_gate_quaternions([q_x, q_x])
        # Result should be ±identity → |w| ≈ 1, imaginary parts ≈ 0
        assert abs(abs(result.w) - 1.0) < 1e-9
        assert abs(result.x) < 1e-9
        assert abs(result.y) < 1e-9
        assert abs(result.z) < 1e-9

    def test_rx_angles_add(self):
        # Rx(a) followed by Rx(b) = Rx(a+b)
        a, b = 0.4, 0.7
        q_a = gate_to_quaternion("Rx", angle=a)
        q_b = gate_to_quaternion("Rx", angle=b)
        result = accumulate_gate_quaternions([q_a, q_b])
        expected = gate_to_quaternion("Rx", angle=a + b)
        assert abs(abs(result.w) - abs(expected.w)) < 1e-10
        assert abs(abs(result.x) - abs(expected.x)) < 1e-10

    def test_result_is_normalised(self):
        qs = [gate_to_quaternion("Rx", angle=0.3)] * 5
        result = accumulate_gate_quaternions(qs)
        norm = math.sqrt(result.w ** 2 + result.x ** 2 + result.y ** 2 + result.z ** 2)
        assert abs(norm - 1.0) < 1e-12

    def test_empty_list_raises(self):
        import pytest
        with pytest.raises(ValueError, match="empty"):
            accumulate_gate_quaternions([])


class TestCanonicalizeGateQuaternion:
    """Tests for canonicalize_gate_quaternion (theory §14)."""

    def test_positive_scalar_part_unchanged(self):
        q = gate_to_quaternion("Rx", angle=0.5)
        c = canonicalize_gate_quaternion(q)
        assert c.w >= 0.0

    def test_negative_scalar_part_flipped(self):
        import rqm_core
        # Construct a quaternion with negative w by negating X (w ≈ 0, x ≈ -1)
        q_x = gate_to_quaternion("X")
        neg = rqm_core.Quaternion(-q_x.w, -q_x.x, -q_x.y, -q_x.z)
        c = canonicalize_gate_quaternion(neg)
        assert c.w >= -1e-12

    def test_canonicalize_is_unit_norm(self):
        q = gate_to_quaternion("Rz", angle=2.5)
        c = canonicalize_gate_quaternion(q)
        norm = math.sqrt(c.w ** 2 + c.x ** 2 + c.y ** 2 + c.z ** 2)
        assert abs(norm - 1.0) < 1e-12

    def test_delegates_to_rqm_core(self):
        import rqm_core
        q = gate_to_quaternion("Ry", angle=1.8)
        c_ours = canonicalize_gate_quaternion(q)
        c_core = q.canonicalize()
        assert abs(c_ours.w - c_core.w) < 1e-15
        assert abs(c_ours.x - c_core.x) < 1e-15
        assert abs(c_ours.y - c_core.y) < 1e-15
        assert abs(c_ours.z - c_core.z) < 1e-15
