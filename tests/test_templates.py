"""
tests/test_templates.py — Tests for rqm_pennylane.templates.
"""

import math

import numpy as np
import pennylane as qml
import pytest

from rqm_pennylane.templates import (
    entangling_layer,
    hardware_efficient_ansatz,
    rqm_angle_embedding,
    single_qubit_layer,
)


@pytest.fixture
def dev1():
    return qml.device("default.qubit", wires=1)


@pytest.fixture
def dev2():
    return qml.device("default.qubit", wires=2)


@pytest.fixture
def dev3():
    return qml.device("default.qubit", wires=3)


class TestSingleQubitLayer:
    def test_returns_finite_expectation(self, dev2):
        @qml.qnode(dev2)
        def circuit(params):
            single_qubit_layer(params, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        params = np.zeros((2, 3))
        result = float(circuit(params))
        assert math.isfinite(result)

    def test_wrong_shape_raises(self, dev2):
        @qml.qnode(dev2)
        def circuit(params):
            single_qubit_layer(params, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="shape"):
            circuit(np.zeros((3, 3)))

    def test_differentiable(self, dev2):
        @qml.qnode(dev2)
        def circuit(params):
            single_qubit_layer(params, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        params = qml.numpy.array(np.zeros((2, 3)), requires_grad=True)
        grads = qml.grad(circuit)(params)
        assert grads.shape == (2, 3)

    def test_identity_params_give_z_one(self, dev2):
        @qml.qnode(dev2)
        def circuit(params):
            single_qubit_layer(params, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        result = float(circuit(np.zeros((2, 3))))
        assert abs(result - 1.0) < 1e-10


class TestEntanglingLayer:
    def test_runs_on_two_qubits(self, dev2):
        @qml.qnode(dev2)
        def circuit(params):
            entangling_layer(params, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        result = float(circuit(np.zeros((2, 3))))
        assert math.isfinite(result)

    def test_runs_on_three_qubits(self, dev3):
        @qml.qnode(dev3)
        def circuit(params):
            entangling_layer(params, wires=[0, 1, 2])
            return qml.expval(qml.PauliZ(0))

        result = float(circuit(np.zeros((3, 3))))
        assert math.isfinite(result)

    def test_single_wire_raises(self, dev1):
        @qml.qnode(dev1)
        def circuit(params):
            entangling_layer(params, wires=[0])
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="at least 2 wires"):
            circuit(np.zeros((1, 3)))

    def test_differentiable(self, dev2):
        @qml.qnode(dev2)
        def circuit(params):
            entangling_layer(params, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        params = qml.numpy.array(np.zeros((2, 3)), requires_grad=True)
        grads = qml.grad(circuit)(params)
        assert grads.shape == (2, 3)


class TestHardwareEfficientAnsatz:
    def test_runs_depth_one(self, dev2):
        @qml.qnode(dev2)
        def circuit(params):
            hardware_efficient_ansatz(params, wires=[0, 1], depth=1)
            return qml.expval(qml.PauliZ(0))

        result = float(circuit(np.zeros((1, 2, 3))))
        assert math.isfinite(result)

    def test_runs_depth_three(self, dev2):
        @qml.qnode(dev2)
        def circuit(params):
            hardware_efficient_ansatz(params, wires=[0, 1], depth=3)
            return qml.expval(qml.PauliZ(0))

        result = float(circuit(np.zeros((3, 2, 3))))
        assert math.isfinite(result)

    def test_wrong_depth_shape_raises(self, dev2):
        @qml.qnode(dev2)
        def circuit(params):
            hardware_efficient_ansatz(params, wires=[0, 1], depth=2)
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="shape"):
            circuit(np.zeros((1, 2, 3)))

    def test_differentiable(self, dev2):
        @qml.qnode(dev2)
        def circuit(params):
            hardware_efficient_ansatz(params, wires=[0, 1], depth=2)
            return qml.expval(qml.PauliZ(0))

        params = qml.numpy.array(np.zeros((2, 2, 3)), requires_grad=True)
        grads = qml.grad(circuit)(params)
        assert grads.shape == (2, 2, 3)


class TestRqmAngleEmbedding:
    def test_runs_on_two_qubits(self, dev2):
        @qml.qnode(dev2)
        def circuit(features):
            rqm_angle_embedding(features, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        result = float(circuit(np.array([0.5, 1.0])))
        assert math.isfinite(result)

    def test_wrong_feature_length_raises(self, dev2):
        @qml.qnode(dev2)
        def circuit(features):
            rqm_angle_embedding(features, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="shape"):
            circuit(np.array([0.5]))

    def test_zero_features_give_z_one(self, dev2):
        @qml.qnode(dev2)
        def circuit(features):
            rqm_angle_embedding(features, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        result = float(circuit(np.array([0.0, 0.0])))
        assert abs(result - 1.0) < 1e-10

    def test_differentiable(self, dev2):
        @qml.qnode(dev2)
        def circuit(features):
            rqm_angle_embedding(features, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        features = qml.numpy.array([0.5, 1.0], requires_grad=True)
        grads = qml.grad(circuit)(features)
        assert grads.shape == (2,)
