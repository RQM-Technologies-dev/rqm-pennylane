"""
examples/basic_quaternion_rotation.py

Apply a quaternion-derived single-qubit rotation and measure expectation values.

Usage
-----
    python examples/basic_quaternion_rotation.py
"""

import math

import pennylane as qml
from pennylane import numpy as pnp

from rqm_pennylane import (
    apply_quaternion_rotation,
    bloch_to_pennylane_state,
    default_qubit_device,
    quaternion_to_rotation_params,
)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
dev = default_qubit_device(wires=1)


# ---------------------------------------------------------------------------
# Circuit: apply a quaternion-derived rotation and measure <X>, <Y>, <Z>
# ---------------------------------------------------------------------------
@qml.qnode(dev)
def rotation_circuit(q):
    """Apply the rotation encoded in unit quaternion *q* and return Pauli expectations."""
    apply_quaternion_rotation(q, wires=0)
    return (
        qml.expval(qml.PauliX(0)),
        qml.expval(qml.PauliY(0)),
        qml.expval(qml.PauliZ(0)),
    )


def run():
    print("=" * 60)
    print("  rqm-pennylane — Basic Quaternion Rotation Example")
    print("=" * 60)

    # Identity quaternion — no rotation, state stays |0⟩
    q_identity = [1.0, 0.0, 0.0, 0.0]
    ex, ey, ez = rotation_circuit(q_identity)
    print(f"\nIdentity rotation (q = {q_identity})")
    print(f"  <X> = {ex:.6f}  <Y> = {ey:.6f}  <Z> = {ez:.6f}")
    print("  Expected: <Z> ≈ 1.0 (no rotation from |0⟩)")

    # 180° rotation around X: (w=0, x=1, y=0, z=0)
    q_x180 = [0.0, 1.0, 0.0, 0.0]
    ex, ey, ez = rotation_circuit(q_x180)
    print(f"\n180° rotation around X (q = {q_x180})")
    print(f"  <X> = {ex:.6f}  <Y> = {ey:.6f}  <Z> = {ez:.6f}")
    print("  Expected: <Z> ≈ -1.0 (state flipped to |1⟩)")

    # 90° rotation around Y: (w=cos(pi/4), x=0, y=sin(pi/4), z=0)
    half_angle = math.pi / 4
    q_y90 = [math.cos(half_angle), 0.0, math.sin(half_angle), 0.0]
    ex, ey, ez = rotation_circuit(q_y90)
    print(f"\n90° rotation around Y")
    print(f"  <X> = {ex:.6f}  <Y> = {ey:.6f}  <Z> = {ez:.6f}")
    print("  Expected: <Z> ≈ 0.0 (state on equator of Bloch sphere)")

    # Show the raw Euler angle conversion
    phi, theta, omega = quaternion_to_rotation_params(q_y90)
    print(f"\n  Quaternion → ZYZ Euler angles:")
    print(f"  phi={phi:.4f}  theta={theta:.4f}  omega={omega:.4f}")

    # Bloch state → statevector
    print("\nBloch north pole → statevector:")
    sv = bloch_to_pennylane_state(theta=0.0, phi=0.0)
    print(f"  {sv}")

    print("\nDone.")


if __name__ == "__main__":
    run()
