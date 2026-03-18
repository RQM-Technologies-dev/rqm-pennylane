"""
templates.py — Reusable variational building blocks for rqm-pennylane.

All templates emit PennyLane-native operations and are differentiable
through PennyLane's autodiff interface.  They are designed to be called
inside QNode-decorated circuits.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import pennylane as qml

__all__ = [
    "single_qubit_layer",
    "entangling_layer",
    "hardware_efficient_ansatz",
    "rqm_angle_embedding",
]

WiresLike = Union[int, str, Sequence[Union[int, str]]]


def single_qubit_layer(
    params: Union[np.ndarray, Sequence],
    wires: Sequence,
) -> None:
    """Apply an independent ``qml.Rot`` gate to each qubit.

    Parameters
    ----------
    params:
        Array of shape ``(len(wires), 3)`` containing the ``(phi, theta, omega)``
        Euler angles for each qubit.
    wires:
        Ordered sequence of qubit wires.

    Raises
    ------
    ValueError
        If *params* does not have shape ``(len(wires), 3)``.

    Example
    -------
    >>> import pennylane as qml, numpy as np
    >>> dev = qml.device("default.qubit", wires=2)
    >>> @qml.qnode(dev)
    ... def circuit(params):
    ...     single_qubit_layer(params, wires=[0, 1])
    ...     return qml.expval(qml.PauliZ(0))
    >>> circuit(np.zeros((2, 3)))
    """
    params = np.asarray(params)
    if params.shape != (len(wires), 3):
        raise ValueError(
            f"params must have shape (len(wires), 3) = ({len(wires)}, 3); "
            f"got {params.shape}."
        )
    for i, wire in enumerate(wires):
        qml.Rot(params[i, 0], params[i, 1], params[i, 2], wires=wire)


def entangling_layer(
    params: Union[np.ndarray, Sequence],
    wires: Sequence,
) -> None:
    """Apply a single-qubit rotation layer followed by a CNOT ring entangler.

    The CNOT ring connects qubit ``i`` to qubit ``(i+1) % n``, creating
    nearest-neighbour entanglement across all qubits.

    Parameters
    ----------
    params:
        Array of shape ``(len(wires), 3)`` containing per-qubit Euler angles.
    wires:
        Ordered sequence of qubit wires (must contain at least 2).

    Raises
    ------
    ValueError
        If fewer than 2 wires are provided or *params* has wrong shape.
    """
    if len(wires) < 2:
        raise ValueError("entangling_layer requires at least 2 wires.")
    single_qubit_layer(params, wires)
    for i in range(len(wires)):
        qml.CNOT(wires=[wires[i], wires[(i + 1) % len(wires)]])


def hardware_efficient_ansatz(
    params: Union[np.ndarray, Sequence],
    wires: Sequence,
    depth: int,
) -> None:
    """Stack *depth* entangling layers to form a hardware-efficient ansatz.

    Parameters
    ----------
    params:
        Array of shape ``(depth, len(wires), 3)`` containing per-layer,
        per-qubit Euler angles.
    wires:
        Ordered sequence of qubit wires (at least 2 for entanglement).
    depth:
        Number of entangling layers to apply.

    Raises
    ------
    ValueError
        If *params* does not have shape ``(depth, len(wires), 3)``.

    Example
    -------
    >>> import pennylane as qml, numpy as np
    >>> dev = qml.device("default.qubit", wires=2)
    >>> @qml.qnode(dev)
    ... def circuit(params):
    ...     hardware_efficient_ansatz(params, wires=[0, 1], depth=2)
    ...     return qml.expval(qml.PauliZ(0))
    >>> circuit(np.zeros((2, 2, 3)))
    """
    params = np.asarray(params)
    if params.shape != (depth, len(wires), 3):
        raise ValueError(
            f"params must have shape (depth, len(wires), 3) = "
            f"({depth}, {len(wires)}, 3); got {params.shape}."
        )
    for d in range(depth):
        entangling_layer(params[d], wires)


def rqm_angle_embedding(
    features: Union[np.ndarray, Sequence],
    wires: Sequence,
) -> None:
    """Encode a feature vector into qubit rotations via angle embedding.

    Each feature value ``f_i`` is encoded as ``RX(f_i)`` on wire ``i``.
    This is the simplest angle embedding and is differentiable with respect
    to *features*.

    Parameters
    ----------
    features:
        1-D array of length ``len(wires)`` containing the feature values.
    wires:
        Ordered sequence of qubit wires.

    Raises
    ------
    ValueError
        If ``len(features) != len(wires)``.

    Example
    -------
    >>> import pennylane as qml, numpy as np
    >>> dev = qml.device("default.qubit", wires=2)
    >>> @qml.qnode(dev)
    ... def circuit(features):
    ...     rqm_angle_embedding(features, wires=[0, 1])
    ...     return qml.expval(qml.PauliZ(0))
    >>> circuit(np.array([0.5, 1.0]))
    """
    features = np.asarray(features)
    if features.shape != (len(wires),):
        raise ValueError(
            f"features must have shape ({len(wires)},); got {features.shape}."
        )
    for i, wire in enumerate(wires):
        qml.RX(features[i], wires=wire)
