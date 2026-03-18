"""
examples/variational_single_qubit.py

Train a tiny single-qubit variational circuit to minimize ⟨Z⟩ (drive the
state toward |1⟩) using gradient descent.

Usage
-----
    python examples/variational_single_qubit.py
"""

import pennylane as qml
from pennylane import numpy as pnp

from rqm_pennylane import (
    default_qubit_device,
    expectation_cost,
    make_variational_qnode,
    optimize_step,
)

# ---------------------------------------------------------------------------
# Device and circuit
# ---------------------------------------------------------------------------
dev = default_qubit_device(wires=1)


def circuit(params):
    """Single-qubit ansatz: Ry then Rz."""
    qml.RY(params[0], wires=0)
    qml.RZ(params[1], wires=0)


qnode = make_variational_qnode(dev, circuit, lambda: qml.expval(qml.PauliZ(0)))

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
STEPS = 20
STEP_SIZE = 0.3


def run():
    print("=" * 60)
    print("  rqm-pennylane — Variational Single-Qubit Training")
    print("=" * 60)
    print(f"\nMinimizing ⟨Z⟩ over {STEPS} gradient descent steps.")
    print(f"Stepsize = {STEP_SIZE}")
    print(f"Optimal solution: ⟨Z⟩ = -1.0  (state |1⟩)\n")

    optimizer = qml.GradientDescentOptimizer(stepsize=STEP_SIZE)
    params = pnp.array([0.01, 0.01], requires_grad=True)

    print(f"{'Step':>5}  {'Cost ⟨Z⟩':>12}  {'params':>30}")
    print("-" * 55)

    for step in range(STEPS):
        cost = expectation_cost(qnode, params)
        print(f"{step:5d}  {cost:12.6f}  {params}")
        params = optimize_step(optimizer, qnode, params)

    final_cost = expectation_cost(qnode, params)
    print("-" * 55)
    print(f"\nFinal cost ⟨Z⟩ = {final_cost:.6f}")
    print(f"Final params    = {params}")
    print("\nDone.")


if __name__ == "__main__":
    run()
