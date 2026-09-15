"""Compiler-descriptor bridge to PennyLane.

The public circuit boundary remains standard RQM descriptors.  Adaptive
relational forms are materialized by rqm-compiler as the smallest standard
RXX/RYY/RZZ sequence before this backend layer.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import pennylane as qml

__all__ = ["compiled_operation_to_pennylane", "compiled_circuit_to_qnode_ops"]

_GATE_MAP: dict[str, tuple[Any, bool]] = {
    "rx": (qml.RX, True),
    "ry": (qml.RY, True),
    "rz": (qml.RZ, True),
    "h": (qml.Hadamard, False),
    "x": (qml.PauliX, False),
    "y": (qml.PauliY, False),
    "z": (qml.PauliZ, False),
    "cnot": (qml.CNOT, False),
    "cx": (qml.CNOT, False),
    "cz": (qml.CZ, False),
    "swap": (qml.SWAP, False),
    "rxx": (qml.IsingXX, True),
    "ryy": (qml.IsingYY, True),
    "rzz": (qml.IsingZZ, True),
}


def compiled_operation_to_pennylane(
    op: Any, wires_override: Sequence[Any] | None = None
) -> Callable[[], qml.operation.Operation]:
    name, wires, params = _extract_op_fields(op)
    if wires_override is not None:
        wires = list(wires_override)
    name_lower = name.lower()
    if name_lower not in _GATE_MAP:
        supported = ", ".join(sorted(_GATE_MAP))
        raise NotImplementedError(f"Gate {name!r} is not supported. Supported gates: {supported}.")
    pl_op, has_params = _GATE_MAP[name_lower]

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
        # Accept both legacy PennyLane bridge shape and canonical rqm-compiler descriptors.
        name = op.get("name", op.get("gate"))
        if not isinstance(name, str):
            raise ValueError("Operation dict must contain 'name' or canonical 'gate'.")
        wires_raw = op.get("wires", op.get("targets"))
        if wires_raw is None:
            raise ValueError("Operation dict must contain 'wires' or canonical 'targets'.")
        wires = list(wires_raw)
        raw_params = op.get("params", [])
        if isinstance(raw_params, dict):
            angle = raw_params.get("angle")
            params = [] if angle is None else [float(angle)]
        else:
            params = [float(value) for value in raw_params]
        return name, wires, params
    try:
        name = str(getattr(op, "name", getattr(op, "gate")))
        wires = list(getattr(op, "wires", getattr(op, "targets")))
        raw_params = getattr(op, "params", [])
    except AttributeError as exc:
        raise ValueError("Operation descriptor is missing gate/name or wires/targets.") from exc
    if isinstance(raw_params, dict):
        angle = raw_params.get("angle")
        params = [] if angle is None else [float(angle)]
    else:
        params = [float(value) for value in raw_params]
    return name, wires, params


def _extract_circuit_operations(circuit: Any) -> list[Any]:
    if isinstance(circuit, dict):
        try:
            return list(circuit["operations"])
        except KeyError as exc:
            raise ValueError("Circuit dict must contain 'operations'.") from exc
    try:
        return list(circuit.operations)
    except AttributeError as exc:
        raise ValueError("Circuit descriptor must have 'operations'.") from exc
