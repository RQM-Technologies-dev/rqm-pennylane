"""
gates.py — Gradient-friendly parameterized gate helpers for rqm-pennylane.

All helpers emit PennyLane-native operations and are fully differentiable
through PennyLane's autodiff interface.  No custom ``Operation`` subclasses
are defined; functions compose existing PennyLane ops.
"""

from __future__ import annotations

from typing import Sequence, Union

import pennylane as qml

from rqm_pennylane.wrappers import quaternion_to_rotation_params

__all__ = [
    "RQMRotation",
    "apply_quaternion_rotation",
    "parameterized_su2",
]

# Type alias accepted by PennyLane wire specifications.
WiresLike = Union[int, str, Sequence[Union[int, str]]]


def RQMRotation(
    phi: float,
    theta: float,
    omega: float,
    wires: WiresLike,
) -> qml.operation.Operation:
    """Apply a one-qubit ZYZ Euler rotation using ``qml.Rot``.

    This is a thin, differentiable wrapper around ``qml.Rot``.  It is
    provided so that callers can use the RQM naming convention and make the
    intent clear when the angles originate from RQM conversion utilities.

    Parameters
    ----------
    phi:
        First ZYZ Euler angle (rotation around Z), in radians.
    theta:
        Second ZYZ Euler angle (rotation around Y), in radians.
    omega:
        Third ZYZ Euler angle (rotation around Z), in radians.
    wires:
        Target qubit wire(s).

    Returns
    -------
    pennylane.operation.Operation
        The applied ``qml.Rot`` operation.

    Example
    -------
    >>> import pennylane as qml
    >>> dev = qml.device("default.qubit", wires=1)
    >>> @qml.qnode(dev)
    ... def circuit(phi, theta, omega):
    ...     RQMRotation(phi, theta, omega, wires=0)
    ...     return qml.expval(qml.PauliZ(0))
    """
    return qml.Rot(phi, theta, omega, wires=wires)


def apply_quaternion_rotation(
    q: Sequence[float],
    wires: WiresLike,
) -> qml.operation.Operation:
    """Apply a single-qubit rotation derived from a unit quaternion.

    The quaternion ``q = (w, x, y, z)`` is converted to ZYZ Euler angles
    via :func:`~rqm_pennylane.wrappers.quaternion_to_rotation_params` and
    then applied as a ``qml.Rot`` gate.

    Parameters
    ----------
    q:
        Length-4 sequence ``(w, x, y, z)`` representing a unit quaternion.
    wires:
        Target qubit wire(s).

    Returns
    -------
    pennylane.operation.Operation
        The applied ``qml.Rot`` operation.

    Example
    -------
    >>> import pennylane as qml
    >>> dev = qml.device("default.qubit", wires=1)
    >>> @qml.qnode(dev)
    ... def circuit():
    ...     apply_quaternion_rotation([1, 0, 0, 0], wires=0)
    ...     return qml.expval(qml.PauliZ(0))
    """
    phi, theta, omega = quaternion_to_rotation_params(q)
    return qml.Rot(phi, theta, omega, wires=wires)


def parameterized_su2(
    alpha: float,
    beta: float,
    gamma: float,
    wires: WiresLike,
) -> None:
    """Apply an SU(2)-equivalent single-qubit decomposition.

    Decomposes the gate as ``Rz(alpha) Ry(beta) Rz(gamma)`` using
    PennyLane primitives.  The result spans the full SU(2) group (up to
    global phase) and is differentiable.

    Parameters
    ----------
    alpha:
        First rotation angle around Z, in radians.
    beta:
        Rotation angle around Y, in radians.
    gamma:
        Second rotation angle around Z, in radians.
    wires:
        Target qubit wire(s).

    Example
    -------
    >>> import pennylane as qml
    >>> dev = qml.device("default.qubit", wires=1)
    >>> @qml.qnode(dev)
    ... def circuit(alpha, beta, gamma):
    ...     parameterized_su2(alpha, beta, gamma, wires=0)
    ...     return qml.expval(qml.PauliZ(0))
    """
    qml.RZ(alpha, wires=wires)
    qml.RY(beta, wires=wires)
    qml.RZ(gamma, wires=wires)
