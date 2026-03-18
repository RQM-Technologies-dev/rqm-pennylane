# rqm-pennylane

**rqm-pennylane** is the PennyLane integration layer for the RQM Quantum Computing ecosystem — differentiable gates, variational helpers, and hybrid quantum-classical workflows.

---

## Ecosystem placement

`rqm-pennylane` is the differentiable / variational entrypoint in the RQM stack.
It consumes canonical math from `rqm-core` and circuit descriptors from
`rqm-compiler`, then exposes them through PennyLane-native operations.

```
                ┌──────────────────┐
                │  rqm-pennylane   │
                │ differentiable   │
                │   workflows      │
                └────────┬─────────┘
                         │
      ┌──────────────────┼──────────────────┐
      │                  │                  │
┌─────┴──────────┐ ┌─────┴──────────┐ ┌────┴────────────┐
│   rqm-core     │ │ rqm-compiler   │ │  rqm-optimize   │
│ canonical math │ │ canonical IR   │ │  circuit form   │
└────────────────┘ └────────────────┘ └─────────────────┘
```

PennyLane is the differentiable / variational entrypoint in the RQM stack.
`rqm-pennylane` does not duplicate math or compiler logic — it bridges them.

---

## Features

- **Wrappers** — convert RQM quaternion, spinor, and Bloch objects to PennyLane-ready values
- **Gradient-friendly gates** — `RQMRotation`, `apply_quaternion_rotation`, `parameterized_su2`
- **Variational templates** — `single_qubit_layer`, `entangling_layer`, `hardware_efficient_ansatz`, `rqm_angle_embedding`
- **Optimization helpers** — `expectation_cost`, `parameter_shift_gradients`, `make_variational_qnode`, `optimize_step`
- **Device utilities** — `default_qubit_device`, `lightning_device`
- **Export bridge** — minimal `rqm-compiler` → PennyLane callable bridge

---

## Installation

```bash
pip install rqm-pennylane
```

For development:

```bash
pip install "rqm-pennylane[dev]"
```

---

## Quickstart

```python
import pennylane as qml
import numpy as np
from rqm_pennylane import (
    default_qubit_device,
    RQMRotation,
    single_qubit_layer,
    expectation_cost,
)

dev = default_qubit_device(wires=1)

@qml.qnode(dev)
def circuit(params):
    RQMRotation(params[0], params[1], params[2], wires=0)
    return qml.expval(qml.PauliZ(0))

params = np.array([0.1, 0.2, 0.3])
print(circuit(params))  # scalar expectation value
```

---

## Architecture rules

1. **Do not** duplicate quaternion / spinor / SU(2) math from `rqm-core`.
2. **Do not** invent a circuit IR; use `rqm-compiler` abstractions.
3. **Do not** add circuit simplification; that belongs in `rqm-optimize`.
4. Focus on PennyLane interoperability and differentiable parameterized workflows.
5. Keep the package backend-agnostic except for PennyLane-specific integration points.
6. Public API must feel lightweight, Pythonic, and useful for researchers.

See [`AGENTS.md`](AGENTS.md) for the full contributor guidelines.

---

## Examples

| File | Description |
|------|-------------|
| [`examples/basic_quaternion_rotation.py`](examples/basic_quaternion_rotation.py) | Apply a quaternion-derived rotation and measure expectation value |
| [`examples/variational_single_qubit.py`](examples/variational_single_qubit.py) | Train a single-qubit variational circuit |
| [`examples/hybrid_cost_example.py`](examples/hybrid_cost_example.py) | Hybrid classical-quantum cost minimization |

Run any example directly:

```bash
python examples/basic_quaternion_rotation.py
```

---

## Initial supported scope (v0.1.0)

### Gates / operations
- `RQMRotation(phi, theta, omega, wires)` — one-qubit Euler rotation
- `apply_quaternion_rotation(q, wires)` — quaternion-derived rotation
- `parameterized_su2(alpha, beta, gamma, wires)` — SU(2)-equivalent decomposition

### Templates
- `single_qubit_layer(params, wires)` — per-qubit Rot layers
- `entangling_layer(params, wires)` — Rot + CNOT ring
- `hardware_efficient_ansatz(params, wires, depth)` — stacked entangling layers
- `rqm_angle_embedding(features, wires)` — angle encoding

### Variational helpers
- `expectation_cost(qnode, params)` — scalar cost from QNode
- `parameter_shift_gradients(qnode, params)` — PennyLane gradient helper
- `make_variational_qnode(device, circuit_fn, measure_fn)` — QNode factory
- `optimize_step(optimizer, cost_fn, params)` — one-step optimization

### Export bridge
- `compiled_operation_to_pennylane(op, wires_override)` — single-op bridge
- `compiled_circuit_to_qnode_ops(compiled_circuit)` — circuit-level bridge

Supported gates in the export bridge: `rx`, `ry`, `rz`, `h`, `x`, `y`, `z`, `cnot`, `cz`, `swap`.

---

## Roadmap

- **v0.2.0** — expanded `rqm-compiler` export coverage, noise model helpers
- **v0.3.0** — optional JAX / Torch backend hints, batched gradient helpers
- **v1.0.0** — stable API, full `rqm-compiler` gate coverage, documentation site

---

## License

MIT — see [`LICENSE`](LICENSE).
