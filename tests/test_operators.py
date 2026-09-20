import math
import unittest

import numpy as np

from tt_obstacle.operators import (
    Q1Operator,
    assemble_p1,
    generalized_eigh,
    uniform_p1,
)
from tt_obstacle.tensor import TensorTrain


class OperatorTests(unittest.TestCase):
    def test_uniform_exact_local_matrices_and_length(self):
        stiffness, mass = uniform_p1(2, length=2)
        np.testing.assert_allclose(stiffness, [[3, -1.5], [-1.5, 3]])
        np.testing.assert_allclose(mass, [[4 / 9, 1 / 9], [1 / 9, 4 / 9]])
        k1, m1 = uniform_p1(1)
        np.testing.assert_allclose(k1, [[4]])
        np.testing.assert_allclose(m1, [[1 / 3]])

    def test_nonuniform_assembly_matches_elementwise_integrals(self):
        nodes = np.array([-1.0, -0.7, -0.1, 0.25, 1.2])
        stiffness, mass = assemble_p1(nodes)
        full_k = np.zeros((len(nodes), len(nodes)))
        full_m = np.zeros_like(full_k)
        for j, h in enumerate(np.diff(nodes)):
            full_k[j:j + 2, j:j + 2] += np.array([[1, -1], [-1, 1]]) / h
            full_m[j:j + 2, j:j + 2] += h * np.array([[2, 1], [1, 2]]) / 6
        np.testing.assert_allclose(stiffness, full_k[1:-1, 1:-1])
        np.testing.assert_allclose(mass, full_m[1:-1, 1:-1])

    def test_generalized_eigenpairs_nonuniform(self):
        stiffness, mass = assemble_p1([0, 0.09, 0.34, 0.61, 0.83, 1])
        values, vectors = generalized_eigh(stiffness, mass)
        np.testing.assert_allclose(vectors.T @ mass @ vectors, np.eye(4), atol=2e-14)
        np.testing.assert_allclose(stiffness @ vectors, (mass @ vectors) * values,
                                   rtol=2e-13, atol=2e-13)
        self.assertTrue(np.all(np.diff(values) > 0))
        self.assertTrue(np.all(vectors[:, 0] > 0))

    def test_contractions_match_dense_with_different_ranks(self):
        rng = np.random.default_rng(1047)
        factors = [assemble_p1([0, 0.2, 0.7, 1]), uniform_p1(3, 2), uniform_p1(2)]
        operator = Q1Operator([x[0] for x in factors], [x[1] for x in factors])
        left_array = rng.normal(size=operator.shape)
        left = TensorTrain.from_dense(left_array)
        right = TensorTrain.rank_one([rng.normal(size=n) for n in operator.shape])
        right_array = right.to_dense()
        left_vector, right_vector = left_array.ravel(), right_array.ravel()
        stiffness = operator.to_dense()
        mass = operator.mass_dense()
        expected = left_vector @ stiffness @ right_vector
        self.assertAlmostEqual(operator.energy_inner(left, right), expected, places=11)
        self.assertAlmostEqual(operator.mass_inner(left, right),
                               left_vector @ mass @ right_vector, places=12)
        self.assertAlmostEqual(operator.mass_norm(left) ** 2,
                               left_vector @ mass @ left_vector, places=11)
        self.assertAlmostEqual(operator.energy_norm(left) ** 2,
                               left_vector @ stiffness @ left_vector, places=10)
        diagonal = np.ones(1)
        for local in operator.diagonal_mass_factors:
            diagonal = np.kron(diagonal, local)
        self.assertAlmostEqual(operator.diagonal_mass_norm(left) ** 2,
                               np.dot(diagonal, left_vector ** 2), places=11)
        np.testing.assert_allclose(stiffness, stiffness.T, atol=1e-15)

    def test_ground_projectors_and_mass_comparison(self):
        factors = [assemble_p1([0, 0.08, 0.33, 0.9, 1]), uniform_p1(4)]
        operator = Q1Operator([x[0] for x in factors], [x[1] for x in factors])
        c = operator.comparison_constant
        self.assertGreater(c, 0)
        self.assertLessEqual(c, 1 + 1e-14)
        for k, (stiffness, mass, b, projector, diagonal) in enumerate(zip(
            operator.stiffness, operator.mass, operator.ground_factors,
            operator.projectors, operator.diagonal_mass_factors
        )):
            np.testing.assert_allclose(projector @ projector, projector, atol=2e-14)
            np.testing.assert_allclose(mass @ projector, projector.T @ mass, atol=1e-14)
            np.testing.assert_allclose(stiffness @ projector, projector.T @ stiffness,
                                       atol=2e-13)
            np.testing.assert_allclose(diagonal * b, mass @ b, atol=1e-14)
            self.assertGreaterEqual(np.linalg.eigvalsh(np.diag(diagonal) - mass)[0],
                                    -1e-13)
            self.assertGreaterEqual(np.linalg.eigvalsh(mass - c * np.diag(diagonal))[0],
                                    -1e-13)
            self.assertTrue(np.all(projector >= 0))
            self.assertAlmostEqual(b @ mass @ b, 1, places=13)
        self.assertAlmostEqual(operator.mass_norm(operator.ground_tensor), 1, places=12)
        self.assertAlmostEqual(operator.energy_norm(operator.ground_tensor) ** 2,
                               sum(operator.ground_eigenvalues), places=10)

    def test_uniform_comparison_and_ground_spectrum(self):
        for n in (1, 2, 5, 12):
            operator = Q1Operator.uniform((n,))
            cosine = math.cos(math.pi / (n + 1))
            expected_c = (2 - cosine) / (2 + cosine)
            self.assertAlmostEqual(operator.comparison_constant, expected_c, places=12)
            self.assertGreaterEqual(operator.comparison_constant, 1 / 3 - 1e-14)

    def test_norms_and_contractions_need_no_dense_tensor(self):
        operator = Q1Operator.uniform((2,) * 18)
        tensor = operator.ground_tensor
        # Default dense matrix conversion would have over 68 billion entries.
        for conversion in (operator.to_dense, operator.mass_dense):
            with self.assertRaises(ValueError):
                conversion()
        self.assertAlmostEqual(operator.mass_norm(tensor), 1, places=12)
        self.assertAlmostEqual(operator.energy_norm(tensor) ** 2,
                               sum(operator.ground_eigenvalues), places=9)
        self.assertAlmostEqual(operator.energy_inner(tensor, tensor),
                               sum(operator.ground_eigenvalues), places=9)

    def test_dense_limit_counts_matrix_entries(self):
        operator = Q1Operator.uniform((3, 2))
        with self.assertRaises(ValueError):
            operator.to_dense(max_entries=35)
        with self.assertRaises(ValueError):
            operator.mass_dense(max_entries=35)
        self.assertEqual(operator.to_dense(max_entries=36).shape, (6, 6))
        with self.assertRaises(ValueError):
            operator.to_dense(max_entries=True)

    def test_whitened_norm_resolves_cancelling_tt_residual(self):
        rng = np.random.default_rng(130)
        operator = Q1Operator.uniform((3, 2, 3, 2))
        base = TensorTrain.rank_one([rng.normal(size=n) for n in operator.shape])
        direction = TensorTrain.rank_one([rng.normal(size=n) for n in operator.shape])
        epsilon = 1e-12
        residual = base.add(direction.scaled(epsilon)).subtract(base)
        for norm in (operator.energy_norm, operator.mass_norm, operator.diagonal_mass_norm):
            expected = epsilon * norm(direction)
            actual = norm(residual)
            self.assertLess(abs(actual - expected), 0.005 * expected)

    def test_spd_energy_is_available_outside_positive_ground_class(self):
        operator = Q1Operator([np.diag([1, 2])], [np.eye(2)])
        tensor = TensorTrain.rank_one([np.array([2., -3.])])
        self.assertAlmostEqual(operator.energy_norm(tensor), math.sqrt(22))
        with self.assertRaises(ValueError):
            _ = operator.ground_factors

    def test_validation_and_input_copy(self):
        for bad in (0, -1, 2.5, True):
            with self.assertRaises(ValueError):
                uniform_p1(bad)
        for bad in ([0, 1], [0, 0.5, 0.5, 1], [0, np.nan, 1]):
            with self.assertRaises(ValueError):
                assemble_p1(bad)
        for shape in ((), (0,), (2.5,)):
            with self.assertRaises(ValueError):
                Q1Operator.uniform(shape)
        with self.assertRaises(ValueError):
            Q1Operator.uniform((2, 2), lengths=(1,))
        with self.assertRaises(ValueError):
            generalized_eigh([[1, 2], [0, 1]], np.eye(2))
        with self.assertRaises(ValueError):
            generalized_eigh([[1, 0], [0, -1]], np.eye(2))
        with self.assertRaises(ValueError):
            Q1Operator([np.eye(2)], [np.eye(3)])
        stiffness, mass = uniform_p1(2)
        operator = Q1Operator([stiffness], [mass])
        stiffness[:] = 0
        self.assertTrue(np.all(np.diag(operator.stiffness[0]) > 0))
        with self.assertRaises(ValueError):
            operator.mass[0][0, 0] = 1
        with self.assertRaises(ValueError):
            operator.energy_norm(TensorTrain.zeros((3,)))


if __name__ == "__main__":
    unittest.main()
