"""
tests/test_wrappers.py — Tests for rqm_pennylane.wrappers.
"""

import math

import numpy as np
import pytest
import rqm_core

from rqm_pennylane.wrappers import (
    bloch_to_pennylane_state,
    quaternion_to_bloch_vector,
    quaternion_to_measurement_probs,
    quaternion_to_rotation_params,
    spinor_to_quaternion_embedding,
    spinor_to_statevector,
)


class TestRqmCoreIntegration:
    """Verify that wrappers.py integrates with the canonical rqm-core library."""

    def test_rqm_core_importable(self):
        assert hasattr(rqm_core, "Quaternion")
        assert hasattr(rqm_core, "normalize_spinor")
        assert hasattr(rqm_core, "bloch_to_state")

    def test_bloch_to_pennylane_state_matches_rqm_core(self):
        theta, phi = 1.0, 0.5
        sv = bloch_to_pennylane_state(theta, phi)
        alpha_ref, beta_ref = rqm_core.bloch_to_state(theta, phi)
        np.testing.assert_allclose(sv[0], alpha_ref, atol=1e-14)
        np.testing.assert_allclose(sv[1], beta_ref, atol=1e-14)

    def test_spinor_to_statevector_matches_rqm_core(self):
        sv = spinor_to_statevector([3 + 4j, 0j])
        alpha_ref, beta_ref = rqm_core.normalize_spinor(3 + 4j, 0j)
        np.testing.assert_allclose(sv[0], alpha_ref, atol=1e-14)
        np.testing.assert_allclose(sv[1], beta_ref, atol=1e-14)

    def test_quaternion_uses_rqm_core_type(self):
        # Verifies that Quaternion from rqm-core normalises correctly
        # and that quaternion_to_rotation_params accepts the same input.
        q_ref = rqm_core.Quaternion(1, 0, 0, 0)
        phi, theta, omega = quaternion_to_rotation_params([q_ref.w, q_ref.x, q_ref.y, q_ref.z])
        assert abs(theta) < 1e-10  # identity → zero rotation



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


class TestSpinorToQuaternionEmbedding:
    """Tests for spinor_to_quaternion_embedding (theory §5, §6)."""

    def test_zero_state_gives_identity_like_quaternion(self):
        # |0⟩ → the quaternion that maps |0⟩ to itself is the identity
        q = spinor_to_quaternion_embedding([1 + 0j, 0j])
        assert abs(q.w) > 0.99
        assert abs(q.x) < 1e-9
        assert abs(q.y) < 1e-9
        assert abs(q.z) < 1e-9

    def test_result_is_unit_quaternion(self):
        # Normalized spinor lives on S³; embedding must be unit norm.
        for spinor in [[1, 0], [0, 1], [1 + 1j, 1 - 1j], [0.6 + 0j, 0.8j]]:
            q = spinor_to_quaternion_embedding(spinor)
            norm = math.sqrt(q.w ** 2 + q.x ** 2 + q.y ** 2 + q.z ** 2)
            assert abs(norm - 1.0) < 1e-12

    def test_matches_rqm_core_spinor_to_quaternion(self):
        import rqm_core
        alpha, beta = 0.6 + 0j, 0.8j
        q = spinor_to_quaternion_embedding([alpha, beta])
        q_ref = rqm_core.spinor_to_quaternion(alpha, beta)
        assert abs(q.w - q_ref.w) < 1e-12
        assert abs(q.x - q_ref.x) < 1e-12
        assert abs(q.y - q_ref.y) < 1e-12
        assert abs(q.z - q_ref.z) < 1e-12

    def test_zero_spinor_raises(self):
        with pytest.raises(ValueError, match="zero norm"):
            spinor_to_quaternion_embedding([0 + 0j, 0j])

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="exactly 2 components"):
            spinor_to_quaternion_embedding([1 + 0j, 0j, 0j])


class TestQuaternionToBlochVector:
    """Tests for quaternion_to_bloch_vector (theory §12)."""

    def test_identity_quaternion_points_to_north_pole(self):
        # Identity gate leaves |0⟩ at the north pole (0, 0, 1).
        bx, by, bz = quaternion_to_bloch_vector([1, 0, 0, 0])
        assert abs(bx) < 1e-12
        assert abs(by) < 1e-12
        assert abs(bz - 1.0) < 1e-12

    def test_x_gate_flips_bloch_vector_to_south_pole(self):
        # X gate quaternion ≈ (0, 1, 0, 0) → rotates |0⟩ to |1⟩ = south pole.
        import rqm_core
        q_x = rqm_core.gate_x()
        bx, by, bz = quaternion_to_bloch_vector([q_x.w, q_x.x, q_x.y, q_x.z])
        assert abs(bz - (-1.0)) < 1e-12

    def test_bloch_vector_is_unit_norm(self):
        for q in [[1, 0, 0, 0], [0, 1, 0, 0], [0.5, 0.5, 0.5, 0.5]]:
            bx, by, bz = quaternion_to_bloch_vector(q)
            norm = math.sqrt(bx ** 2 + by ** 2 + bz ** 2)
            assert abs(norm - 1.0) < 1e-10

    def test_unnormalized_quaternion_accepted(self):
        # Scaling a quaternion by a positive constant should give the same Bloch vector.
        bx1, by1, bz1 = quaternion_to_bloch_vector([1, 0, 0, 0])
        bx2, by2, bz2 = quaternion_to_bloch_vector([2, 0, 0, 0])
        assert abs(bz1 - bz2) < 1e-12

    def test_zero_quaternion_raises(self):
        with pytest.raises(ValueError, match="zero norm"):
            quaternion_to_bloch_vector([0, 0, 0, 0])

    def test_matches_rqm_core_bloch_from_quaternion(self):
        import rqm_core
        q_s = rqm_core.gate_s()
        bx, by, bz = quaternion_to_bloch_vector([q_s.w, q_s.x, q_s.y, q_s.z])
        bx_ref, by_ref, bz_ref = rqm_core.bloch_from_quaternion(q_s)
        assert abs(bx - bx_ref) < 1e-12
        assert abs(by - by_ref) < 1e-12
        assert abs(bz - bz_ref) < 1e-12


class TestQuaternionToMeasurementProbs:
    """Tests for quaternion_to_measurement_probs (theory §17)."""

    def test_identity_state_prob_zero_is_one(self):
        # |0⟩ → P(0) = 1, P(1) = 0
        p0, p1 = quaternion_to_measurement_probs([1, 0, 0, 0])
        assert abs(p0 - 1.0) < 1e-12
        assert abs(p1 - 0.0) < 1e-12

    def test_x_gate_state_prob_one_is_one(self):
        # X|0⟩ = |1⟩ → P(0) = 0, P(1) = 1
        import rqm_core
        q_x = rqm_core.gate_x()
        p0, p1 = quaternion_to_measurement_probs([q_x.w, q_x.x, q_x.y, q_x.z])
        assert abs(p0 - 0.0) < 1e-12
        assert abs(p1 - 1.0) < 1e-12

    def test_hadamard_state_gives_equal_probs(self):
        # H|0⟩ → P(0) = P(1) = 0.5
        import rqm_core
        q_h = rqm_core.gate_h()
        p0, p1 = quaternion_to_measurement_probs([q_h.w, q_h.x, q_h.y, q_h.z])
        assert abs(p0 - 0.5) < 1e-12
        assert abs(p1 - 0.5) < 1e-12

    def test_probs_sum_to_one(self):
        for q in [[1, 0, 0, 0], [0.5, 0.5, 0.5, 0.5], [0, 1, 0, 0]]:
            p0, p1 = quaternion_to_measurement_probs(q)
            assert abs(p0 + p1 - 1.0) < 1e-12

    def test_probs_in_unit_interval(self):
        for q in [[1, 0, 0, 0], [0.5, 0.5, 0.5, 0.5], [0, 0, 1, 0]]:
            p0, p1 = quaternion_to_measurement_probs(q)
            assert 0.0 <= p0 <= 1.0 + 1e-12
            assert 0.0 <= p1 <= 1.0 + 1e-12

    def test_zero_quaternion_raises(self):
        with pytest.raises(ValueError, match="zero norm"):
            quaternion_to_measurement_probs([0, 0, 0, 0])
