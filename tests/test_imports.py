"""
tests/test_imports.py — Verify that the public API imports cleanly.
"""

import rqm_pennylane


def test_version():
    assert hasattr(rqm_pennylane, "__version__")
    assert isinstance(rqm_pennylane.__version__, str)


def test_all_symbols_importable():
    for name in rqm_pennylane.__all__:
        assert hasattr(rqm_pennylane, name), f"Missing public symbol: {name}"


def test_wrappers_importable():
    from rqm_pennylane import (
        bloch_to_pennylane_state,
        quaternion_to_rotation_params,
        spinor_to_statevector,
    )
    assert callable(quaternion_to_rotation_params)
    assert callable(spinor_to_statevector)
    assert callable(bloch_to_pennylane_state)


def test_gates_importable():
    from rqm_pennylane import (
        RQMRotation,
        apply_quaternion_rotation,
        parameterized_su2,
    )
    assert callable(RQMRotation)
    assert callable(apply_quaternion_rotation)
    assert callable(parameterized_su2)


def test_templates_importable():
    from rqm_pennylane import (
        entangling_layer,
        hardware_efficient_ansatz,
        rqm_angle_embedding,
        single_qubit_layer,
    )
    assert callable(single_qubit_layer)
    assert callable(entangling_layer)
    assert callable(hardware_efficient_ansatz)
    assert callable(rqm_angle_embedding)


def test_variational_importable():
    from rqm_pennylane import (
        expectation_cost,
        make_variational_qnode,
        optimize_step,
        parameter_shift_gradients,
    )
    assert callable(expectation_cost)
    assert callable(parameter_shift_gradients)
    assert callable(make_variational_qnode)
    assert callable(optimize_step)


def test_devices_importable():
    from rqm_pennylane import default_qubit_device, lightning_device
    assert callable(default_qubit_device)
    assert callable(lightning_device)


def test_export_importable():
    from rqm_pennylane import (
        compiled_circuit_to_qnode_ops,
        compiled_operation_to_pennylane,
    )
    assert callable(compiled_operation_to_pennylane)
    assert callable(compiled_circuit_to_qnode_ops)
