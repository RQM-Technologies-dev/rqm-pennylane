"""
examples/hybrid_cost_example.py

A small hybrid classical-quantum cost workflow.

The quantum part encodes two classical features as qubit rotations and
measures ⟨Z₀ Z₁⟩.  The classical part adds a quadratic regularisation term.
A PennyLane gradient-descent optimizer drives both toward a minimum.

Usage
-----
    python examples/hybrid_cost_example.py
"""

import math

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from rqm_pennylane import (
    default_qubit_device,
    hardware_efficient_ansatz,
    optimize_step,
)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
N_WIRES = 2
DEPTH = 2
dev = default_qubit_device(wires=N_WIRES)


# ---------------------------------------------------------------------------
# Quantum circuit
# ---------------------------------------------------------------------------
@qml.qnode(dev)
def quantum_circuit(params):
    """Hardware-efficient ansatz returning the ZZ correlation."""
    hardware_efficient_ansatz(params, wires=list(range(N_WIRES)), depth=DEPTH)
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


# ---------------------------------------------------------------------------
# Hybrid cost function: quantum expectation + L2 regularisation
# ---------------------------------------------------------------------------
LAMBDA = 0.05  # regularisation strength


def hybrid_cost(params):
    """Quantum expectation value plus L2 regularisation."""
    q_cost = quantum_circuit(params)
    l2_reg = LAMBDA * pnp.sum(params ** 2)
    return q_cost + l2_reg


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
STEPS = 15
STEP_SIZE = 0.2


def run():
    print("=" * 60)
    print("  rqm-pennylane — Hybrid Classical-Quantum Cost Example")
    print("=" * 60)
    print(f"\nCircuit: {N_WIRES} qubits, depth {DEPTH} hardware-efficient ansatz")
    print(f"Cost = ⟨Z₀Z₁⟩ + {LAMBDA} · ‖params‖²")
    print(f"Optimizer: GradientDescent, stepsize={STEP_SIZE}, steps={STEPS}\n")

    rng = np.random.default_rng(42)
    params = pnp.array(
        rng.uniform(-math.pi, math.pi, size=(DEPTH, N_WIRES, 3)),
        requires_grad=True,
    )

    optimizer = qml.GradientDescentOptimizer(stepsize=STEP_SIZE)

    print(f"{'Step':>5}  {'Hybrid cost':>14}  {'⟨Z₀Z₁⟩':>10}")
    print("-" * 38)

    for step in range(STEPS):
        cost = float(hybrid_cost(params))
        qzz = float(quantum_circuit(params))
        print(f"{step:5d}  {cost:14.6f}  {qzz:10.6f}")
        params = optimize_step(optimizer, hybrid_cost, params)

    final_cost = float(hybrid_cost(params))
    final_qzz = float(quantum_circuit(params))
    print("-" * 38)
    print(f"\nFinal hybrid cost = {final_cost:.6f}")
    print(f"Final ⟨Z₀Z₁⟩     = {final_qzz:.6f}")
    print("\nDone.")


if __name__ == "__main__":
    run()
