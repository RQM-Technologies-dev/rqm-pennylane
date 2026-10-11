"""Compiler-descriptor bridge to PennyLane.

Adaptive relational forms are materialized by rqm-compiler as the smallest
standard RXX/RYY/RZZ sequence before this backend layer.
"""
from __future__ import annotations
from typing import Any, Callable, Sequence
import math
import pennylane as qml

__all__ = ["compiled_operation_to_pennylane", "compiled_circuit_to_qnode_ops", "pennylane_operations_to_compiler"]
_GATE_MAP: dict[str, tuple[Any, bool]] = {
    "rx": (qml.RX, True), "ry": (qml.RY, True), "rz": (qml.RZ, True),
    "h": (qml.Hadamard, False), "x": (qml.PauliX, False),
    "y": (qml.PauliY, False), "z": (qml.PauliZ, False),
    "cnot": (qml.CNOT, False), "cx": (qml.CNOT, False),
    "cz": (qml.CZ, False), "swap": (qml.SWAP, False),
    "rxx": (qml.IsingXX, True), "ryy": (qml.IsingYY, True),
    "rzz": (qml.IsingZZ, True),
}

def pennylane_operations_to_compiler(operations: Sequence[Any], *, wire_order: Sequence[Any]) -> Any:
    """Export bound unitary operations into the optional canonical compiler Circuit.

    ``wire_order`` explicitly defines logical qubit indices, including idle wires.
    This is a serialization boundary, not an autodiff operation. Measurements,
    observables, shots and experiment reductions must be supplied separately to
    the hardware planner; a QuantumScript is deliberately not accepted because
    silently discarding its measurements would change the requested experiment.
    No execution, decomposition or optimization occurs here.
    """
    try:
        from rqm_compiler import Circuit, Operation
    except ImportError as exc:
        raise ImportError("Install rqm-compiler to export PennyLane operations.") from exc
    if isinstance(operations, qml.tape.QuantumScript):
        raise ValueError("Export explicit operations and preserve measurements separately.")
    wires = list(wire_order)
    if not wires or len(set(wires)) != len(wires):
        raise ValueError("wire_order must contain unique wires and include idle wires.")
    indices = {wire: index for index, wire in enumerate(wires)}
    # Match exact native classes: custom subclasses cannot silently claim a
    # familiar gate name while implementing different semantics.
    supported = {native: (gate, angle) for gate, (native, angle) in _GATE_MAP.items()}
    supported[qml.CNOT] = ("cx", False)
    circuit = Circuit(len(wires))
    for op in operations:
        if type(op) not in supported:
            raise NotImplementedError(f"Unsupported PennyLane operation: {type(op).__name__}")
        gate, parameterized = supported[type(op)]
        try:
            mapped = [indices[wire] for wire in op.wires]
        except KeyError as exc:
            raise ValueError("Operation wire is absent from wire_order.") from exc
        if len(mapped) != len(set(mapped)) or len(op.data) != int(parameterized):
            raise ValueError("Invalid operation wires or parameter count.")
        params = {}
        if parameterized:
            value = op.data[0]
            if qml.math.is_abstract(value) or qml.math.ndim(value) != 0:
                raise ValueError("Bind scalar parameters before hardware export; batches need a sweep.")
            angle = float(value)
            if not math.isfinite(angle):
                raise ValueError("Hardware parameters must be finite.")
            params = {"angle": angle}
        controls, targets = (mapped[:1], mapped[1:]) if gate in {"cx", "cz"} else ([], mapped)
        circuit.add(Operation(gate=gate, targets=targets, controls=controls, params=params))
    return circuit

def compiled_operation_to_pennylane(op: Any, wires_override: Sequence[Any] | None = None) -> Callable[[], qml.operation.Operation]:
    name, wires, params = _extract_op_fields(op)
    if wires_override is not None:
        wires = list(wires_override)
    key = name.lower()
    if key not in _GATE_MAP:
        raise NotImplementedError(f"Gate {name!r} is not supported. Supported gates: {', '.join(sorted(_GATE_MAP))}.")
    pl_op, has_params = _GATE_MAP[key]
    def apply() -> qml.operation.Operation:
        if has_params:
            if not params:
                raise ValueError(f"Gate {name!r} requires an angle parameter.")
            return pl_op(*params, wires=wires)
        return pl_op(wires=wires)
    return apply

def compiled_circuit_to_qnode_ops(compiled_circuit: Any) -> list[Callable[[], qml.operation.Operation]]:
    return [compiled_operation_to_pennylane(op) for op in _extract_circuit_operations(compiled_circuit)]

def _extract_op_fields(op: Any) -> tuple[str, list[Any], list[float]]:
    if isinstance(op, dict):
        name = op.get("name") if "name" in op else op.get("gate")
        wires_raw = op.get("wires") if "wires" in op else op.get("targets")
        raw_params = op.get("params", [])
        controls = op.get("controls", [])
    else:
        name = getattr(op, "name", None)
        if name is None:
            name = getattr(op, "gate", None)
        wires_raw = getattr(op, "wires", None)
        if wires_raw is None:
            wires_raw = getattr(op, "targets", None)
        raw_params = getattr(op, "params", [])
        controls = getattr(op, "controls", [])
    if not isinstance(name, str) or wires_raw is None:
        raise ValueError("Operation descriptor is missing gate/name or wires/targets.")
    wires = list(wires_raw)
    if controls:
        if name.lower() not in {"cx", "cnot", "cz"} or len(controls) != 1 or len(wires) != 1:
            raise NotImplementedError("Unsupported explicit control descriptor.")
        wires = list(controls) + wires
    if isinstance(raw_params, dict):
        angle = raw_params.get("angle")
        params = [] if angle is None else [float(angle)]
    else:
        params = [float(value) for value in raw_params]
    return name, wires, params

def _extract_circuit_operations(circuit: Any) -> list[Any]:
    if isinstance(circuit, dict):
        if "operations" not in circuit:
            raise ValueError("Circuit dict must contain 'operations'.")
        return list(circuit["operations"])
    operations = getattr(circuit, "operations", None)
    if operations is None:
        raise ValueError("Circuit descriptor must have 'operations'.")
    return list(operations)
