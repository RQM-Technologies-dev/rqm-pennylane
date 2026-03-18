"""
wrappers.py — Bridge helpers from RQM objects to PennyLane-ready values.

These functions accept quaternion-like, spinor-like, or Bloch-sphere inputs
and return plain Python / NumPy values that PennyLane gates and QNodes can
consume directly.

Canonical quaternion, spinor, and Bloch sphere math is provided by
``rqm-core``.  This module imports from there and adds only the
PennyLane-specific output conversions (e.g. numpy array wrapping).

The sole local numeric computation retained here is the ZYZ Euler-angle
decomposition in ``quaternion_to_rotation_params``.  ``rqm-core`` provides
the ``Quaternion`` type and SU(2) matrix conversion, but does not expose a
ZYZ decomposition API; that decomposition is specific to PennyLane's
``qml.Rot`` gate convention and therefore belongs in this adapter layer.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np
from rqm_core import Quaternion, bloch_to_state, normalize_spinor

__all__ = [
    "quaternion_to_rotation_params",
    "spinor_to_statevector",
    "bloch_to_pennylane_state",
]


def quaternion_to_rotation_params(
    q: Sequence[float],
) -> Tuple[float, float, float]:
    """Convert a unit quaternion to ZYZ Euler angles suitable for ``qml.Rot``.

    Uses ``rqm_core.Quaternion`` for canonical quaternion construction and
    normalisation.  The ZYZ decomposition is computed locally because it is
    specific to PennyLane's ``qml.Rot(phi, theta, omega)`` gate convention.

    Parameters
    ----------
    q:
        A length-4 sequence ``(w, x, y, z)`` representing a unit quaternion.
        The quaternion is normalised internally before conversion.

    Returns
    -------
    tuple[float, float, float]
        ``(phi, theta, omega)`` Euler angles in radians, compatible with
        ``qml.Rot(phi, theta, omega, wires=...)``.

    Raises
    ------
    ValueError
        If *q* has zero norm.

    Notes
    -----
    The decomposition follows the standard ZYZ convention:
    ``R = Rz(phi) Ry(theta) Rz(omega)``.
    """
    w, x, y, z = (float(v) for v in q)

    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        raise ValueError("Quaternion has zero norm; cannot convert to rotation angles.")

    # Delegate normalisation to rqm-core's canonical Quaternion type.
    quat = Quaternion(w / norm, x / norm, y / norm, z / norm)
    w, x, y, z = quat.w, quat.x, quat.y, quat.z

    # ZYZ Euler decomposition from the SU(2) matrix:
    #   U = [[w + iz,  y + ix],
    #        [-y + ix, w - iz]]
    # phi = atan2(z, w) + atan2(x, y)
    # omega = atan2(z, w) - atan2(x, y)
    # theta = 2 * atan2(sqrt(x² + y²), sqrt(w² + z²))
    # This decomposition is PennyLane-specific and lives in this adapter.
    phi = math.atan2(z, w) + math.atan2(x, y)
    omega = math.atan2(z, w) - math.atan2(x, y)
    theta = 2.0 * math.atan2(math.sqrt(x * x + y * y), math.sqrt(w * w + z * z))

    return phi, theta, omega


def spinor_to_statevector(
    spinor: Sequence[complex],
) -> np.ndarray:
    """Normalise a two-component spinor into a PennyLane-compatible statevector.

    Delegates normalisation to ``rqm_core.normalize_spinor`` and wraps the
    result as a NumPy array for direct use with PennyLane.

    Parameters
    ----------
    spinor:
        A length-2 sequence of complex amplitudes ``(alpha, beta)``.

    Returns
    -------
    numpy.ndarray
        A normalised complex array of shape ``(2,)`` with ``dtype=complex128``.

    Raises
    ------
    ValueError
        If *spinor* does not have exactly 2 elements or has zero norm.
    """
    arr = np.asarray(spinor, dtype=complex)
    if arr.shape != (2,):
        raise ValueError(
            f"A spinor must have exactly 2 components; got shape {arr.shape}."
        )
    try:
        alpha, beta = normalize_spinor(complex(arr[0]), complex(arr[1]))
    except ValueError as exc:
        raise ValueError(
            "Spinor has zero norm; cannot produce a valid statevector."
        ) from exc
    return np.array([alpha, beta], dtype=complex)


def bloch_to_pennylane_state(
    theta: float,
    phi: float,
) -> np.ndarray:
    """Convert Bloch sphere coordinates to a one-qubit statevector.

    Delegates to ``rqm_core.bloch_to_state`` for canonical Bloch-sphere
    conversion and wraps the result as a NumPy array for PennyLane.

    The Bloch sphere parametrisation is:
    ``|ψ⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩``

    Parameters
    ----------
    theta:
        Polar angle in radians (0 ≤ θ ≤ π).
    phi:
        Azimuthal angle in radians (0 ≤ φ < 2π).

    Returns
    -------
    numpy.ndarray
        Normalised complex statevector of shape ``(2,)`` with ``dtype=complex128``.
    """
    alpha, beta = bloch_to_state(theta, phi)
    return np.array([alpha, beta], dtype=complex)
