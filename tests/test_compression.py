import unittest
from unittest.mock import patch

import numpy as np

from tt_obstacle.compression import feasible_compress
from tt_obstacle.operators import Q1Operator
from tt_obstacle.tensor import TensorTrain


class CompressionTests(unittest.TestCase):
    def test_mass_whitened_matrix_rounding_and_unmet_cap(self):
        operator = Q1Operator.uniform((3, 3))
        dense = np.array([[1.0, -2.0, 0.3], [0.2, 2.0, 1.3], [-1.0, 0.7, 3.0]])
        original = TensorTrain.from_dense(dense)
        result = feasible_compress(original, operator, rtol=1e-12, max_rank=1, method="core")
        w0, w1 = [np.linalg.cholesky(m).T for m in operator.mass]
        weighted = w0 @ dense @ w1.T
        u, singular, vt = np.linalg.svd(weighted, full_matrices=False)
        best = singular[0] * np.outer(u[:, 0], vt[0])
        expected = np.linalg.solve(w0, best) @ np.linalg.inv(w1.T)
        np.testing.assert_allclose(result.candidate.to_dense(), expected, atol=2e-12, rtol=2e-12)
        self.assertAlmostEqual(result.rounding_mass_error, np.linalg.norm(weighted - best), places=12)
        self.assertFalse(result.rounding_tolerance_met)
        self.assertEqual(result.rounding_ranks, (1,))
        self.assertGreaterEqual(np.min(result.tensor.to_dense() - dense), -2e-12)

    def test_zero_and_full_accuracy_rounding(self):
        operator = Q1Operator.uniform((2, 3, 2))
        zero = TensorTrain.zeros(operator.shape)
        result = feasible_compress(zero, operator, rtol=0.0)
        self.assertEqual(result.tensor.norm(), 0.0)
        self.assertTrue(result.rounding_tolerance_met)
        original = TensorTrain.from_dense(np.random.default_rng(79).normal(size=operator.shape))
        result = feasible_compress(original, operator, rtol=1e-10)
        self.assertTrue(result.rounding_tolerance_met)
        actual = operator.energy_norm(result.tensor.subtract(original))
        self.assertLessEqual(actual, result.error_bound + 1e-10)

    def test_degree_projection_and_controlled_repair(self):
        operator = Q1Operator.uniform((2, 2, 2))
        original = operator.ground_tensor
        for k in range(3):
            factors = list(operator.ground_factors)
            factors[k] = operator.eigenvectors[k][:, 1]
            original = original.add(TensorTrain.rank_one(factors).scaled(0.15 * (k + 1)))
        result = feasible_compress(
            original, operator, rtol=0.0, max_rank=1, degree=1,
            method="controlled_rank_one", kappa=1.0, structure_certified=True,
        )
        self.assertEqual(result.rounding_ranks, (1, 1))
        self.assertTrue(result.assumptions)
        self.assertGreaterEqual(
            np.min(result.tensor.to_dense() - original.to_dense()), -5e-12
        )
        # Direct coefficient transform independently checks the missing pair/triple modes.
        coefficients = result.candidate.to_dense()
        for axis, (vectors, mass) in enumerate(zip(operator.eigenvectors, operator.mass)):
            coefficients = np.tensordot(vectors.T @ mass, coefficients, axes=(1, axis))
            coefficients = np.moveaxis(coefficients, 0, axis)
        for index in np.ndindex(coefficients.shape):
            if sum(index) > 1:
                self.assertLess(abs(coefficients[index]), 2e-12)

    def test_refuses_original_outside_assumed_degree(self):
        operator = Q1Operator.uniform((2, 2))
        original = TensorTrain.rank_one([e[:, 1] for e in operator.eigenvectors])
        with self.assertRaisesRegex(ValueError, "interaction-degree"):
            feasible_compress(original, operator, degree=1)

    def test_invalid_tolerance_and_missing_structural_assertion(self):
        operator = Q1Operator.uniform((2, 2))
        original = operator.ground_tensor
        for tolerance in (-1.0, float("nan"), float("inf"), True):
            with self.subTest(tolerance=tolerance):
                with self.assertRaises(ValueError):
                    feasible_compress(original, operator, rtol=tolerance)
        with self.assertRaises(ValueError):
            feasible_compress(original, operator, method="controlled_rank_one")
        with self.assertRaises(ValueError):
            feasible_compress(
                original, operator, method="controlled_rank_one", degree=0, kappa=1.0
            )

    def test_no_dense_calls_in_compression(self):
        operator = Q1Operator.uniform((3, 3, 3))
        original = operator.ground_tensor.add(
            TensorTrain.rank_one([np.array([0.1, -0.2, 0.3])] * 3)
        )
        with patch.object(TensorTrain, "to_dense", side_effect=AssertionError("dense call")):
            result = feasible_compress(original, operator, rtol=1e-5, max_rank=1)
        self.assertGreaterEqual(result.error_bound, 0.0)


if __name__ == "__main__":
    unittest.main()
