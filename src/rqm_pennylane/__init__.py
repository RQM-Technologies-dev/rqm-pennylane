"""
rqm_pennylane
=============

Differentiable and hybrid quantum workflows for the RQM ecosystem, built on
PennyLane and rqm-core.

Provides:
- Wrappers that convert RQM objects (quaternions, spinors, Bloch vectors)
  into PennyLane-compatible representations, backed by rqm-core math.
- Gradient-friendly parameterized gate helpers.
- Reusable variational building-block templates.
- Optimization step utilities.
- Device convenience constructors.
- A minimal bridge from rqm-compiler circuit descriptors to PennyLane callables.

Architectural rule: this package is a thin adapter layer. It imports canonical
quaternion / spinor / SU(2) math from rqm-core and circuit descriptors from
rqm-compiler. It does not reimplement that math or circuit optimization.
"""

from rqm_pennylane.devices import default_qubit_device, lightning_device
from rqm_pennylane.export import (
    compiled_circuit_to_qnode_ops,
    compiled_operation_to_pennylane,
)
from rqm_pennylane.gates import (
    RQMRotation,
    accumulate_gate_quaternions,
    apply_quaternion_rotation,
    canonicalize_gate_quaternion,
    gate_to_quaternion,
    parameterized_su2,
)
from rqm_pennylane.templates import (
    entangling_layer,
    hardware_efficient_ansatz,
    rqm_angle_embedding,
    single_qubit_layer,
)
from rqm_pennylane.variational import (
    expectation_cost,
    make_variational_qnode,
    optimize_step,
    parameter_shift_gradients,
)
from rqm_pennylane.wrappers import (
    bloch_to_pennylane_state,
    quaternion_to_bloch_vector,
    quaternion_to_measurement_probs,
    quaternion_to_rotation_params,
    spinor_to_quaternion_embedding,
    spinor_to_statevector,
)

__version__ = "0.1.0"

__all__ = [
    # wrappers
    "quaternion_to_rotation_params",
    "spinor_to_statevector",
    "bloch_to_pennylane_state",
    "spinor_to_quaternion_embedding",
    "quaternion_to_bloch_vector",
    "quaternion_to_measurement_probs",
    # gates
    "RQMRotation",
    "apply_quaternion_rotation",
    "parameterized_su2",
    "gate_to_quaternion",
    "accumulate_gate_quaternions",
    "canonicalize_gate_quaternion",
    # templates
    "single_qubit_layer",
    "entangling_layer",
    "hardware_efficient_ansatz",
    "rqm_angle_embedding",
    # variational
    "expectation_cost",
    "parameter_shift_gradients",
    "make_variational_qnode",
    "optimize_step",
    # devices
    "default_qubit_device",
    "lightning_device",
    # export
    "compiled_operation_to_pennylane",
    "compiled_circuit_to_qnode_ops",
    # version
    "__version__",
]
