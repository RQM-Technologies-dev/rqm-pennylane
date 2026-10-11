"""Canonical export preserves unitary semantics and rejects implicit workloads."""
import numpy as np
import pennylane as qml
import pytest

from rqm_pennylane import pennylane_operations_to_compiler, compiled_circuit_to_qnode_ops

pytest.importorskip("rqm_compiler")


def test_export_roundtrip_preserves_controls_endian_and_idle_wires():
    wires = ["idle", "target", "control"]
    ops = [qml.Hadamard("control"), qml.CNOT(["control", "target"]),
           qml.RY(0.71, "target"), qml.CZ(["target", "idle"]),
           qml.IsingXX(-0.43, ["control", "idle"])]
    circuit = pennylane_operations_to_compiler(ops, wire_order=wires)
    assert circuit.num_qubits == 3
    assert circuit.operations[1].controls == [2]
    assert circuit.operations[1].targets == [1]
    with qml.tape.QuantumTape() as restored:
        for apply in compiled_circuit_to_qnode_ops(circuit):
            apply()
    np.testing.assert_allclose(
        qml.matrix(qml.tape.QuantumScript(ops), wire_order=wires),
        qml.matrix(restored, wire_order=[0, 1, 2]), atol=1e-12,
    )


@pytest.mark.parametrize("op", [qml.RX(np.nan, 0), qml.RY(np.inf, 0),
                                  qml.RZ(np.array([0.1, 0.2]), 0)])
def test_unbound_or_nonfinite_parameters_rejected(op):
    with pytest.raises(ValueError):
        pennylane_operations_to_compiler([op], wire_order=[0])


@pytest.mark.parametrize("op", [qml.AmplitudeDamping(0.1, 0),
                                  qml.StatePrep([1, 0], 0), qml.sample(wires=0)])
def test_nonunitary_and_measurements_rejected(op):
    with pytest.raises(NotImplementedError):
        pennylane_operations_to_compiler([op], wire_order=[0])


def test_script_cannot_silently_lose_readout():
    script = qml.tape.QuantumScript([qml.Hadamard(0)], [qml.expval(qml.PauliX(0))])
    with pytest.raises(ValueError, match="measurements separately"):
        pennylane_operations_to_compiler(script, wire_order=[0])


@pytest.mark.parametrize("wires", [[], [0, 0], [1]])
def test_wire_mapping_must_be_explicit_and_complete(wires):
    with pytest.raises(ValueError):
        pennylane_operations_to_compiler([qml.Hadamard(0)], wire_order=wires)


def test_unknown_control_is_not_dropped():
    with pytest.raises(NotImplementedError, match="control descriptor"):
        compiled_circuit_to_qnode_ops({"operations": [
            {"gate": "rx", "targets": [1], "controls": [0], "params": {"angle": 0.3}}
        ]})
