"""
devices.py — Convenience helpers for constructing PennyLane device instances.
"""

from __future__ import annotations

import pennylane as qml

__all__ = [
    "default_qubit_device",
    "lightning_device",
]


def default_qubit_device(wires: int) -> qml.devices.Device:
    """Create a ``default.qubit`` device with the specified number of wires.

    Parameters
    ----------
    wires:
        Number of qubits.

    Returns
    -------
    pennylane.devices.Device
        A PennyLane ``default.qubit`` device instance.

    Example
    -------
    >>> from rqm_pennylane import default_qubit_device
    >>> dev = default_qubit_device(wires=2)
    """
    return qml.device("default.qubit", wires=wires)


def lightning_device(wires: int) -> qml.devices.Device:
    """Create a ``lightning.qubit`` device with the specified number of wires.

    ``lightning.qubit`` is a high-performance C++ state-vector simulator
    provided by the ``pennylane-lightning`` package.  If it is not installed
    this function raises a clear ``ImportError``.

    Parameters
    ----------
    wires:
        Number of qubits.

    Returns
    -------
    pennylane.devices.Device
        A PennyLane ``lightning.qubit`` device instance.

    Raises
    ------
    ImportError
        If ``pennylane-lightning`` is not installed.

    Example
    -------
    >>> from rqm_pennylane import lightning_device
    >>> dev = lightning_device(wires=2)  # requires pennylane-lightning
    """
    try:
        return qml.device("lightning.qubit", wires=wires)
    except (qml.DeviceError, Exception) as exc:
        raise ImportError(
            "The 'lightning.qubit' device requires the 'pennylane-lightning' package. "
            "Install it with:  pip install pennylane-lightning"
        ) from exc
