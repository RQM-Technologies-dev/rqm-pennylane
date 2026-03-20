"""
gates.py — Gradient-friendly parameterized gate helpers for rqm-pennylane.

All helpers emit PennyLane-native operations and are fully differentiable
through PennyLane's autodiff interface.  No custom ``Operation`` subclasses
are defined; functions compose existing PennyLane ops.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import pennylane as qml
from rqm_core import Quaternion
from rqm_core import gate_h as _gate_h
from rqm_core import gate_identity as _gate_identity
from rqm_core import gate_rx as _gate_rx
from rqm_core import gate_ry as _gate_ry
from rqm_core import gate_rz as _gate_rz
from rqm_core import gate_s as _gate_s
from rqm_core import gate_t as _gate_t
from rqm_core import gate_x as _gate_x
from rqm_core import gate_y as _gate_y
from rqm_core import gate_z as _gate_z

from rqm_pennylane.wrappers import quaternion_to_rotation_params

__all__ = [
    "RQMRotation",
    "apply_quaternion_rotation",
    "parameterized_su2",
    "gate_to_quaternion",
    "accumulate_gate_quaternions",
    "canonicalize_gate_quaternion",
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


# ---------------------------------------------------------------------------
# Named-gate quaternion helpers  (theory §§9-14)
# ---------------------------------------------------------------------------

# Maps lower-case gate name → no-arg factory that returns a unit Quaternion.
# Parameterised gates (rx, ry, rz) require an explicit angle argument.
_FIXED_GATE_QUATERNION_MAP = {
    "i": _gate_identity,
    "identity": _gate_identity,
    "x": _gate_x,
    "y": _gate_y,
    "z": _gate_z,
    "h": _gate_h,
    "s": _gate_s,
    "t": _gate_t,
}

_PARAMETERIZED_GATE_QUATERNION_MAP = {
    "rx": _gate_rx,
    "ry": _gate_ry,
    "rz": _gate_rz,
}


def gate_to_quaternion(
    name: str,
    angle: Optional[float] = None,
) -> Quaternion:
    """Return the unit quaternion representing a standard single-qubit gate.

    Every single-qubit SU(2) gate is exactly representable as a unit
    quaternion ``q = cos(θ/2) + u·sin(θ/2)`` where ``u`` is the rotation
    axis and ``θ`` is the physical Bloch-space rotation angle (theory §9).

    Delegates to the corresponding ``rqm_core.gate_*`` factory; no math is
    performed locally.

    Parameters
    ----------
    name:
        Gate name (case-insensitive).  Fixed (non-parameterised) gates:
        ``"I"``, ``"identity"``, ``"X"``, ``"Y"``, ``"Z"``, ``"H"``,
        ``"S"``, ``"T"``.  Parameterised gates: ``"Rx"``, ``"Ry"``,
        ``"Rz"`` (require *angle*).
    angle:
        Rotation angle in radians.  Required for ``Rx``, ``Ry``, ``Rz``;
        ignored for fixed gates.

    Returns
    -------
    rqm_core.Quaternion
        Unit quaternion for the named gate.

    Raises
    ------
    ValueError
        If a parameterised gate is requested without supplying *angle*, or
        if *name* is not a recognised gate.

    Examples
    --------
    >>> q_x = gate_to_quaternion("X")
    >>> q_rx = gate_to_quaternion("Rx", angle=1.5)
    """
    key = name.lower()
    if key in _FIXED_GATE_QUATERNION_MAP:
        return _FIXED_GATE_QUATERNION_MAP[key]()
    if key in _PARAMETERIZED_GATE_QUATERNION_MAP:
        if angle is None:
            raise ValueError(
                f"Gate '{name}' is parameterised; please supply an 'angle' value."
            )
        return _PARAMETERIZED_GATE_QUATERNION_MAP[key](float(angle))
    supported = sorted(set(_FIXED_GATE_QUATERNION_MAP) | set(_PARAMETERIZED_GATE_QUATERNION_MAP))
    raise ValueError(
        f"Gate '{name}' is not a recognised single-qubit gate.  "
        f"Supported gates: {supported}."
    )


def accumulate_gate_quaternions(
    quaternions: List[Quaternion],
) -> Quaternion:
    """Fuse a sequence of single-qubit gate quaternions into one.

    Quaternion multiplication corresponds to gate composition (theory §10).
    Gates are applied left-to-right: ``quaternions[0]`` is the first gate
    applied to the qubit and ``quaternions[-1]`` is the last.  Following the
    standard matrix convention where the total unitary is the *right-to-left*
    product of matrices, the accumulated quaternion is built as::

        result = quaternions[-1] · … · quaternions[1] · quaternions[0]

    That is, each successive gate quaternion left-multiplies the running
    result, so ``q_later * q_earlier`` in Hamilton-product order.

    The result is normalised to guard against accumulated floating-point
    drift, as required by the theory (§21).

    Parameters
    ----------
    quaternions:
        Ordered sequence of unit quaternions, one per gate.  The first
        element corresponds to the first gate applied to the qubit.

    Returns
    -------
    rqm_core.Quaternion
        Accumulated, normalised unit quaternion representing the combined
        single-qubit operation.

    Raises
    ------
    ValueError
        If *quaternions* is empty.

    Notes
    -----
    This function operates only on *single-qubit* SU(2) segments.  Do not
    pass quaternions across entangling gate boundaries (theory §18, §19).
    """
    if not quaternions:
        raise ValueError(
            "Cannot accumulate an empty list of gate quaternions.  "
            "Provide at least one quaternion."
        )
    result = quaternions[0]
    for q in quaternions[1:]:
        result = q * result
    return result.normalize()


def canonicalize_gate_quaternion(q: Quaternion) -> Quaternion:
    """Return the canonical shortest-geodesic representative of a gate quaternion.

    Both ``q`` and ``-q`` represent the same physical rotation in SO(3).
    For compiler canonicalization the convention ``w ≥ 0`` is chosen so
    that each rotation has a unique shortest-geodesic representative on
    the unit 3-sphere S³ (theory §14).

    Delegates to ``rqm_core.Quaternion.canonicalize``; no math is performed
    locally.

    Parameters
    ----------
    q:
        A unit (or nearly unit) quaternion representing a single-qubit gate.

    Returns
    -------
    rqm_core.Quaternion
        Normalised quaternion with non-negative scalar part (``w ≥ 0``).
    """
    return q.canonicalize()
