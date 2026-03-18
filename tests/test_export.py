"""
tests/test_export.py — Tests for rqm_pennylane.export.
"""

import types

import numpy as np
import pennylane as qml
import pytest

from rqm_pennylane.export import (
    compiled_circuit_to_qnode_ops,
    compiled_operation_to_pennylane,
)


def make_op(name, wires, params=None):
    """Helper: create a simple namespace-based operation descriptor."""
    return types.SimpleNamespace(name=name, wires=wires, params=params or [])


def make_circuit(ops):
    """Helper: create a simple circuit descriptor."""
    return types.SimpleNamespace(operations=ops)


class TestCompiledOperationToPennylane:
    def test_returns_callable(self):
        op = make_op("h", [0])
        fn = compiled_operation_to_pennylane(op)
        assert callable(fn)

    def test_unsupported_gate_raises(self):
        op = make_op("toffoli", [0, 1, 2])
        with pytest.raises(NotImplementedError, match="toffoli"):
            compiled_operation_to_pennylane(op)

    def test_missing_fields_raises(self):
        with pytest.raises(ValueError):
            compiled_operation_to_pennylane(types.SimpleNamespace(name="h"))

    def test_dict_descriptor_accepted(self):
        op = {"name": "h", "wires": [0], "params": []}
        fn = compiled_operation_to_pennylane(op)
        assert callable(fn)

    def test_wires_override(self):
        op = make_op("h", [0])
        fn = compiled_operation_to_pennylane(op, wires_override=[1])
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            fn()
            return qml.expval(qml.PauliZ(1))

        result = float(circuit())
        assert isinstance(result, float)

    @pytest.mark.parametrize("gate_name", ["h", "x", "y", "z"])
    def test_fixed_single_qubit_gates_execute(self, gate_name):
        dev = qml.device("default.qubit", wires=1)
        op = make_op(gate_name, [0])
        fn = compiled_operation_to_pennylane(op)

        @qml.qnode(dev)
        def circuit():
            fn()
            return qml.expval(qml.PauliZ(0))

        result = float(circuit())
        assert isinstance(result, float)

    @pytest.mark.parametrize("gate_name", ["rx", "ry", "rz"])
    def test_parameterized_single_qubit_gates_execute(self, gate_name):
        dev = qml.device("default.qubit", wires=1)
        op = make_op(gate_name, [0], [0.5])
        fn = compiled_operation_to_pennylane(op)

        @qml.qnode(dev)
        def circuit():
            fn()
            return qml.expval(qml.PauliZ(0))

        result = float(circuit())
        assert isinstance(result, float)

    @pytest.mark.parametrize("gate_name", ["cnot", "cz", "swap"])
    def test_two_qubit_gates_execute(self, gate_name):
        dev = qml.device("default.qubit", wires=2)
        op = make_op(gate_name, [0, 1])
        fn = compiled_operation_to_pennylane(op)

        @qml.qnode(dev)
        def circuit():
            fn()
            return qml.expval(qml.PauliZ(0))

        result = float(circuit())
        assert isinstance(result, float)

    def test_parameterized_gate_missing_params_raises(self):
        dev = qml.device("default.qubit", wires=1)
        op = make_op("rx", [0], [])  # no params for a parameterised gate
        fn = compiled_operation_to_pennylane(op)

        @qml.qnode(dev)
        def circuit():
            fn()
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="parameter"):
            circuit()

    def test_case_insensitive_gate_name(self):
        dev = qml.device("default.qubit", wires=1)
        op = make_op("H", [0])
        fn = compiled_operation_to_pennylane(op)

        @qml.qnode(dev)
        def circuit():
            fn()
            return qml.expval(qml.PauliZ(0))

        result = float(circuit())
        assert isinstance(result, float)


class TestCompiledCircuitToQnodeOps:
    def test_empty_circuit_returns_empty_list(self):
        circuit = make_circuit([])
        fns = compiled_circuit_to_qnode_ops(circuit)
        assert fns == []

    def test_single_op_circuit(self):
        circuit = make_circuit([make_op("h", [0])])
        fns = compiled_circuit_to_qnode_ops(circuit)
        assert len(fns) == 1
        assert callable(fns[0])

    def test_multi_op_circuit_executes(self):
        dev = qml.device("default.qubit", wires=2)
        ops = [
            make_op("h", [0]),
            make_op("cnot", [0, 1]),
            make_op("rz", [1], [0.3]),
        ]
        circuit = make_circuit(ops)
        fns = compiled_circuit_to_qnode_ops(circuit)

        @qml.qnode(dev)
        def run():
            for fn in fns:
                fn()
            return qml.expval(qml.PauliZ(0))

        result = float(run())
        assert isinstance(result, float)

    def test_dict_circuit_accepted(self):
        circuit_dict = {
            "operations": [{"name": "h", "wires": [0], "params": []}]
        }
        fns = compiled_circuit_to_qnode_ops(circuit_dict)
        assert len(fns) == 1

    def test_missing_operations_field_raises(self):
        with pytest.raises(ValueError, match="operations"):
            compiled_circuit_to_qnode_ops(types.SimpleNamespace())

    def test_unsupported_gate_raises(self):
        circuit = make_circuit([make_op("toffoli", [0, 1, 2])])
        with pytest.raises(NotImplementedError):
            compiled_circuit_to_qnode_ops(circuit)
