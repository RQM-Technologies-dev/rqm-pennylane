"""
tests/test_wrappers.py — Tests for rqm_pennylane.wrappers.
"""

import math

import numpy as np
import pytest

from rqm_pennylane.wrappers import (
    bloch_to_pennylane_state,
    quaternion_to_rotation_params,
    spinor_to_statevector,
)


class TestQuaternionToRotationParams:
    def test_identity_quaternion(self):
        phi, theta, omega = quaternion_to_rotation_params([1, 0, 0, 0])
        assert math.isfinite(phi)
        assert math.isfinite(theta)
        assert math.isfinite(omega)
        assert abs(theta) < 1e-10  # identity → zero rotation angle

    def test_returns_three_floats(self):
        result = quaternion_to_rotation_params([1, 0, 0, 0])
        assert len(result) == 3
        for v in result:
            assert isinstance(v, float)

    def test_normalises_input(self):
        # Quaternion [2, 0, 0, 0] is equivalent to [1, 0, 0, 0].
        phi1, theta1, omega1 = quaternion_to_rotation_params([1, 0, 0, 0])
        phi2, theta2, omega2 = quaternion_to_rotation_params([2, 0, 0, 0])
        assert abs(theta1 - theta2) < 1e-10

    def test_180_degree_rotation_around_x(self):
        # Quaternion for 180° around X: (w=0, x=1, y=0, z=0)
        phi, theta, omega = quaternion_to_rotation_params([0, 1, 0, 0])
        # After ZYZ decomposition, theta should be pi.
        assert abs(theta - math.pi) < 1e-10

    def test_zero_quaternion_raises(self):
        with pytest.raises(ValueError, match="zero norm"):
            quaternion_to_rotation_params([0, 0, 0, 0])

    def test_output_is_finite_for_arbitrary_quaternion(self):
        q = [0.5, 0.5, 0.5, 0.5]
        phi, theta, omega = quaternion_to_rotation_params(q)
        assert math.isfinite(phi)
        assert math.isfinite(theta)
        assert math.isfinite(omega)


class TestSpinorToStatevector:
    def test_normalized_output(self):
        sv = spinor_to_statevector([1, 0])
        assert abs(np.linalg.norm(sv) - 1.0) < 1e-12

    def test_complex_spinor_normalized(self):
        sv = spinor_to_statevector([1 + 1j, 1 - 1j])
        assert abs(np.linalg.norm(sv) - 1.0) < 1e-12

    def test_output_dtype(self):
        sv = spinor_to_statevector([1, 0])
        assert sv.dtype == complex

    def test_output_shape(self):
        sv = spinor_to_statevector([0, 1])
        assert sv.shape == (2,)

    def test_zero_spinor_raises(self):
        with pytest.raises(ValueError, match="zero norm"):
            spinor_to_statevector([0, 0])

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="exactly 2 components"):
            spinor_to_statevector([1, 0, 0])

    def test_known_state_up(self):
        sv = spinor_to_statevector([1, 0])
        np.testing.assert_allclose(sv, [1 + 0j, 0 + 0j], atol=1e-12)

    def test_known_state_down(self):
        sv = spinor_to_statevector([0, 1])
        np.testing.assert_allclose(sv, [0 + 0j, 1 + 0j], atol=1e-12)


class TestBlochToPennylaneState:
    def test_north_pole_is_state_zero(self):
        sv = bloch_to_pennylane_state(theta=0.0, phi=0.0)
        np.testing.assert_allclose(sv, [1 + 0j, 0 + 0j], atol=1e-12)

    def test_south_pole_is_state_one(self):
        sv = bloch_to_pennylane_state(theta=math.pi, phi=0.0)
        np.testing.assert_allclose(np.abs(sv), [0.0, 1.0], atol=1e-12)

    def test_equator_normalized(self):
        sv = bloch_to_pennylane_state(theta=math.pi / 2, phi=0.0)
        assert abs(np.linalg.norm(sv) - 1.0) < 1e-12

    def test_output_shape_and_dtype(self):
        sv = bloch_to_pennylane_state(0.5, 1.0)
        assert sv.shape == (2,)
        assert sv.dtype == complex

    def test_arbitrary_point_normalized(self):
        for theta in [0.1, 0.5, 1.0, 2.0, 3.0]:
            for phi in [0.0, 0.3, 1.5, 3.0]:
                sv = bloch_to_pennylane_state(theta, phi)
                assert abs(np.linalg.norm(sv) - 1.0) < 1e-12
