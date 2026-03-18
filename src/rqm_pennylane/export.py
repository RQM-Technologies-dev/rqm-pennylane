"""
export.py — Minimal bridge from rqm-compiler operation descriptors to PennyLane.

This module translates a small initial subset of standard single-qubit and
basic two-qubit gate descriptors (as produced by rqm-compiler) into callable
PennyLane operations.

Scope for v0.1.0
----------------
Supported gate names (case-insensitive):
  Single-qubit: rx, ry, rz, h, x, y, z
  Two-qubit:    cnot, cz, swap

Unsupported operations raise ``NotImplementedError`` with a clear message.
The goal is to create the first bridge, not to translate every possible gate.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import pennylane as qml

__all__ = [
    "compiled_operation_to_pennylane",
    "compiled_circuit_to_qnode_ops",
]

# ---------------------------------------------------------------------------
# Internal gate map
# ---------------------------------------------------------------------------

# Maps lower-case gate name → (pennylane_op, has_params)
# has_params=True  → the op is parameterised; params passed as first arg(s)
# has_params=False → the op is fixed; no numeric params
_GATE_MAP: Dict[str, Tuple[Any, bool]] = {
    "rx": (qml.RX, True),
    "ry": (qml.RY, True),
    "rz": (qml.RZ, True),
    "h": (qml.Hadamard, False),
    "x": (qml.PauliX, False),
    "y": (qml.PauliY, False),
    "z": (qml.PauliZ, False),
    "cnot": (qml.CNOT, False),
    "cz": (qml.CZ, False),
    "swap": (qml.SWAP, False),
}


def compiled_operation_to_pennylane(
    op: Any,
    wires_override: Optional[Sequence] = None,
) -> Callable[[], qml.operation.Operation]:
    """Convert a single rqm-compiler operation descriptor to a PennyLane callable.

    The operation descriptor *op* is expected to expose:
    - ``op.name``   (str)  — gate name (case-insensitive)
    - ``op.wires``  (list) — target qubits
    - ``op.params`` (list) — numeric parameters (may be empty)

    Plain dict-like objects with the same keys are also accepted.

    Parameters
    ----------
    op:
        An rqm-compiler operation descriptor or a compatible dict with
        ``name``, ``wires``, and ``params`` keys.
    wires_override:
        If provided, replaces ``op.wires`` with this wire specification.

    Returns
    -------
    Callable[[], pennylane.operation.Operation]
        A zero-argument callable that, when invoked inside a QNode, applies
        the corresponding PennyLane operation and returns it.

    Raises
    ------
    NotImplementedError
        If the gate name is not in the supported subset.
    ValueError
        If the operation descriptor is missing required fields.

    Example
    -------
    >>> import types
    >>> op = types.SimpleNamespace(name="rx", wires=[0], params=[0.5])
    >>> fn = compiled_operation_to_pennylane(op)
    """
    name, wires, params = _extract_op_fields(op)
    if wires_override is not None:
        wires = list(wires_override)

    name_lower = name.lower()
    if name_lower not in _GATE_MAP:
        supported = ", ".join(sorted(_GATE_MAP))
        raise NotImplementedError(
            f"Gate '{name}' is not supported by rqm-pennylane v0.1.0. "
            f"Supported gates: {supported}. "
            "Full gate coverage is planned for a future release."
        )

    pl_op, has_params = _GATE_MAP[name_lower]

    def apply() -> qml.operation.Operation:
        if has_params:
            if not params:
                raise ValueError(
                    f"Gate '{name}' requires at least one parameter but none were provided."
                )
            if len(params) == 1:
                return pl_op(params[0], wires=wires)
            return pl_op(*params, wires=wires)
        return pl_op(wires=wires)

    return apply


def compiled_circuit_to_qnode_ops(
    compiled_circuit: Any,
) -> List[Callable[[], qml.operation.Operation]]:
    """Convert an rqm-compiler circuit descriptor into a list of PennyLane callables.

    The circuit descriptor *compiled_circuit* is expected to expose an
    ``operations`` attribute (or key) that is a sequence of operation
    descriptors compatible with :func:`compiled_operation_to_pennylane`.

    Parameters
    ----------
    compiled_circuit:
        An rqm-compiler circuit descriptor or a compatible object with an
        ``operations`` attribute/key.

    Returns
    -------
    list[Callable]
        Ordered list of zero-argument callables that apply PennyLane ops
        when invoked inside a QNode.

    Raises
    ------
    ValueError
        If *compiled_circuit* does not expose an ``operations`` field.
    NotImplementedError
        If any operation in the circuit uses an unsupported gate.

    Example
    -------
    >>> import types
    >>> op = types.SimpleNamespace(name="h", wires=[0], params=[])
    >>> circuit = types.SimpleNamespace(operations=[op])
    >>> fns = compiled_circuit_to_qnode_ops(circuit)
    >>> # Inside a QNode: [fn() for fn in fns]
    """
    ops = _extract_circuit_operations(compiled_circuit)
    return [compiled_operation_to_pennylane(op) for op in ops]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_op_fields(op: Any) -> Tuple[str, List, List]:
    """Extract (name, wires, params) from an operation descriptor or dict."""
    if isinstance(op, dict):
        try:
            name = op["name"]
            wires = list(op["wires"])
            params = list(op.get("params", []))
        except KeyError as exc:
            raise ValueError(
                "Operation dict must contain 'name' and 'wires' keys."
            ) from exc
    else:
        try:
            name = op.name
            wires = list(op.wires)
            params = list(getattr(op, "params", []))
        except AttributeError as exc:
            raise ValueError(
                "Operation descriptor must have 'name' and 'wires' attributes."
            ) from exc
    return name, wires, params


def _extract_circuit_operations(circuit: Any) -> List[Any]:
    """Extract the operations list from a circuit descriptor or dict."""
    if isinstance(circuit, dict):
        try:
            return list(circuit["operations"])
        except KeyError as exc:
            raise ValueError(
                "Circuit dict must contain an 'operations' key."
            ) from exc
    try:
        return list(circuit.operations)
    except AttributeError as exc:
        raise ValueError(
            "Circuit descriptor must have an 'operations' attribute."
        ) from exc
