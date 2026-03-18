# AGENTS.md — rqm-pennylane

## Role of this repository

`rqm-pennylane` is an **adapter layer** that connects the RQM Quantum Computing
ecosystem to [PennyLane](https://pennylane.ai/) for differentiable and variational
quantum workflows. It is **not** a math library, compiler, or optimizer.

---

## Architectural rules for contributors and agents

### 1. Do not duplicate `rqm-core` math
All quaternion, spinor, SU(2), and Bloch sphere math lives in `rqm-core`.
Import from there. Do not reimplement it here.

### 2. Do not build a compiler or circuit IR
`rqm-compiler` owns the canonical circuit descriptor. This repo may consume
its output (via `export.py`) but must never define competing abstractions.

### 3. Do not add circuit simplification logic
That belongs in `rqm-optimize`. Keep this repo focused on PennyLane execution.

### 4. Focus on PennyLane interoperability
Every module here should have a clear PennyLane integration story:
- `wrappers.py` — converts RQM objects to PennyLane-ready values
- `gates.py` — emits PennyLane-native operations
- `templates.py` — reusable variational building blocks (QNode-compatible)
- `variational.py` — optimization helpers using PennyLane tools
- `devices.py` — convenience device constructors
- `export.py` — bridge from `rqm-compiler` to PennyLane

### 5. Keep the public API small and stable
Only symbols listed in `__all__` in `src/rqm_pennylane/__init__.py` are public.
Think carefully before adding to it. Shrinking `__all__` is a breaking change.

### 6. Prioritize differentiability and usability
All gate helpers and templates must be differentiable through PennyLane's
autodiff interface. Use `qml.Rot`, `qml.RX`, `qml.RY`, `qml.RZ`, `qml.CNOT`,
and other PennyLane-native ops wherever possible. Avoid custom `Operation`
subclasses unless there is a compelling reason.

### 7. Handle optional dependencies gracefully
`rqm-compiler` is an optional dependency for `export.py`. If it is not
installed, raise a clear `ImportError` with an actionable message. Do not
let optional imports leak into the main package namespace.

### 8. No placeholder code
`NotImplementedError` is acceptable for explicitly unsupported export paths.
Dead code, `pass` stubs, and TODO-spam are not acceptable.

---

## What belongs here (checklist)

- [x] Thin wrappers from RQM types to PennyLane-compatible values
- [x] Parameterized gate helpers that emit PennyLane ops
- [x] Reusable variational templates compatible with QNodes
- [x] Optimization step helpers using PennyLane optimizers
- [x] Device convenience constructors
- [x] A minimal `rqm-compiler` → PennyLane bridge (first subset only)

## What does NOT belong here

- Quaternion / spinor / SU(2) math implementations
- A full circuit translator for every possible gate
- Machine-learning framework integrations (torch, tf, jax) in the base install
- General-purpose circuit optimization
- A new circuit IR or descriptor format
