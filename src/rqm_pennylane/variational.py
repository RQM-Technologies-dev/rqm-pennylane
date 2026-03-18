"""
variational.py — Optimization helpers for variational quantum workflows.

These utilities wrap PennyLane's differentiation and optimizer interfaces
to provide a small, practical API for quantum-classical optimization loops.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence, Union

import numpy as np
import pennylane as qml

__all__ = [
    "expectation_cost",
    "parameter_shift_gradients",
    "make_variational_qnode",
    "optimize_step",
]


def expectation_cost(
    qnode: qml.QNode,
    params: Union[np.ndarray, Sequence],
) -> float:
    """Evaluate a scalar cost from a QNode and parameters.

    The *qnode* must return a scalar (e.g. ``qml.expval``).  The result is
    cast to a plain Python float for convenience.

    Parameters
    ----------
    qnode:
        A compiled ``qml.QNode`` that accepts *params* as its sole argument.
    params:
        Parameter array passed to *qnode*.

    Returns
    -------
    float
        The scalar expectation value returned by the QNode.

    Example
    -------
    >>> import pennylane as qml, numpy as np
    >>> dev = qml.device("default.qubit", wires=1)
    >>> @qml.qnode(dev)
    ... def circuit(params):
    ...     qml.RY(params[0], wires=0)
    ...     return qml.expval(qml.PauliZ(0))
    >>> expectation_cost(circuit, np.array([0.5]))
    """
    return float(qnode(params))


def parameter_shift_gradients(
    qnode: qml.QNode,
    params: np.ndarray,
) -> np.ndarray:
    """Return gradients of *qnode* with respect to *params* using PennyLane.

    PennyLane's default differentiation method (parameter-shift when the
    device supports it, else finite differences) is used.

    Parameters
    ----------
    qnode:
        A compiled ``qml.QNode``.
    params:
        NumPy array of parameters; must be a ``pennylane.numpy`` array or
        a plain NumPy array (converted internally).

    Returns
    -------
    numpy.ndarray
        Gradient array with the same shape as *params*.

    Example
    -------
    >>> import pennylane as qml
    >>> from pennylane import numpy as pnp
    >>> dev = qml.device("default.qubit", wires=1)
    >>> @qml.qnode(dev)
    ... def circuit(params):
    ...     qml.RY(params[0], wires=0)
    ...     return qml.expval(qml.PauliZ(0))
    >>> grads = parameter_shift_gradients(circuit, pnp.array([0.5], requires_grad=True))
    """
    grad_fn = qml.grad(qnode)
    pnp_params = qml.numpy.array(np.asarray(params), requires_grad=True)
    return np.asarray(grad_fn(pnp_params))


def make_variational_qnode(
    device: qml.devices.Device,
    circuit_fn: Callable[..., Any],
    measure_fn: Callable[[], Any],
) -> qml.QNode:
    """Create a QNode for a variational workflow.

    Parameters
    ----------
    device:
        A PennyLane device instance (e.g. from :func:`~rqm_pennylane.devices.default_qubit_device`).
    circuit_fn:
        A callable that applies gate operations.  It receives whatever
        arguments the caller passes to the returned QNode.
    measure_fn:
        A zero-argument callable that returns one or more PennyLane
        measurement processes (e.g. ``lambda: qml.expval(qml.PauliZ(0))``).

    Returns
    -------
    qml.QNode
        A compiled, differentiable QNode.

    Example
    -------
    >>> import pennylane as qml, numpy as np
    >>> from rqm_pennylane import default_qubit_device, make_variational_qnode
    >>> dev = default_qubit_device(wires=1)
    >>> def circuit(params):
    ...     qml.RY(params[0], wires=0)
    >>> qnode = make_variational_qnode(dev, circuit, lambda: qml.expval(qml.PauliZ(0)))
    """

    @qml.qnode(device)
    def _qnode(*args: Any, **kwargs: Any) -> Any:
        circuit_fn(*args, **kwargs)
        return measure_fn()

    return _qnode


def optimize_step(
    optimizer: Any,
    cost_fn: Callable[..., float],
    params: np.ndarray,
) -> np.ndarray:
    """Perform one optimizer step and return updated parameters.

    Parameters
    ----------
    optimizer:
        A PennyLane optimizer instance (e.g. ``qml.GradientDescentOptimizer``).
    cost_fn:
        A callable that accepts *params* and returns a scalar cost value.
    params:
        Current parameter array.

    Returns
    -------
    numpy.ndarray
        Updated parameter array after one optimization step.

    Example
    -------
    >>> import pennylane as qml
    >>> from pennylane import numpy as pnp
    >>> dev = qml.device("default.qubit", wires=1)
    >>> @qml.qnode(dev)
    ... def circuit(params):
    ...     qml.RY(params[0], wires=0)
    ...     return qml.expval(qml.PauliZ(0))
    >>> opt = qml.GradientDescentOptimizer(stepsize=0.1)
    >>> params = pnp.array([0.5], requires_grad=True)
    >>> params = optimize_step(opt, circuit, params)
    """
    updated_params, _ = optimizer.step_and_cost(cost_fn, params)
    return updated_params
