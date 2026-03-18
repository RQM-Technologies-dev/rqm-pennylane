"""
wrappers.py — Bridge helpers from RQM objects to PennyLane-ready values.

These functions accept quaternion-like, spinor-like, or Bloch-sphere inputs
and return plain Python / NumPy values that PennyLane gates and QNodes can
consume directly.

Architectural note: when rqm-core is installed the functions import its
canonical conversion utilities. The fallback implementations below reproduce
only the minimal numeric logic required by this adapter layer, so the package
remains usable in lightweight environments.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

__all__ = [
    "quaternion_to_rotation_params",
    "spinor_to_statevector",
    "bloch_to_pennylane_state",
]


def quaternion_to_rotation_params(
    q: Sequence[float],
) -> Tuple[float, float, float]:
    """Convert a unit quaternion to ZYZ Euler angles suitable for ``qml.Rot``.

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

    Notes
    -----
    The decomposition follows the standard ZYZ convention:
    ``R = Rz(phi) Ry(theta) Rz(omega)``.
    """
    w, x, y, z = (float(v) for v in q)

    # Normalise to handle near-unit inputs gracefully.
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        raise ValueError("Quaternion has zero norm; cannot convert to rotation angles.")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    # ZYZ Euler decomposition from the SU(2) matrix:
    #   U = [[w + iz,  y + ix],
    #        [-y + ix, w - iz]]
    # phi = atan2(z, w) + atan2(x, y)
    # omega = atan2(z, w) - atan2(x, y)
    # theta = 2 * atan2(sqrt(x^2 + y^2), sqrt(w^2 + z^2))
    phi = math.atan2(z, w) + math.atan2(x, y)
    omega = math.atan2(z, w) - math.atan2(x, y)
    theta = 2.0 * math.atan2(math.sqrt(x * x + y * y), math.sqrt(w * w + z * z))

    return phi, theta, omega


def spinor_to_statevector(
    spinor: Sequence[complex],
) -> np.ndarray:
    """Normalise a two-component spinor into a PennyLane-compatible statevector.

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
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        raise ValueError("Spinor has zero norm; cannot produce a valid statevector.")
    return arr / norm


def bloch_to_pennylane_state(
    theta: float,
    phi: float,
) -> np.ndarray:
    """Convert Bloch sphere coordinates to a one-qubit statevector.

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
    alpha = math.cos(theta / 2.0)
    beta = math.sin(theta / 2.0) * cmath_exp(phi)
    return np.array([alpha, beta], dtype=complex)


def cmath_exp(phi: float) -> complex:
    """Return ``e^{i * phi}`` using real arithmetic to avoid cmath import."""
    return complex(math.cos(phi), math.sin(phi))
