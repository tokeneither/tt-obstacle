import unittest
from unittest.mock import patch

import numpy as np

from tt_obstacle.operators import Q1Operator
from tt_obstacle.repair import (
    controlled_rank_one_envelope,
    core_envelope,
    quadratic_envelope,
    repair_feasible,
)
from tt_obstacle.tensor import TensorTrain


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.operator = Q1Operator.uniform((3, 2, 3))
        self.rng = np.random.default_rng(371)

    def test_quadratic_formula_signed_tensor_and_rank_bound(self):
        dense = self.rng.normal(size=self.operator.shape)
        residual = TensorTrain.from_dense(dense)
        envelope = quadratic_envelope(residual, self.operator)
        sigma = self.operator.diagonal_mass_norm(residual)
        ground = self.operator.ground_tensor.to_dense()
        expected = (dense * dense / ground + sigma * sigma * ground) / (2 * sigma)
        np.testing.assert_allclose(envelope.to_dense(), expected, rtol=2e-13, atol=2e-13)
        self.assertGreaterEqual(np.min(envelope.to_dense() - np.abs(dense)), -2e-12)
        for actual, rank in zip(envelope.ranks, residual.ranks):
            self.assertLessEqual(actual, 1 + rank * (rank + 1) // 2)

    def test_core_envelope_is_unconditional_rank_one(self):
        for shape in [(5,), (2, 3), (2, 3, 2, 2)]:
            dense = self.rng.normal(size=shape)
            residual = TensorTrain.from_dense(dense)
            envelope = core_envelope(residual)
            self.assertTrue(all(rank == 1 for rank in envelope.ranks))
            self.assertGreaterEqual(np.min(envelope.to_dense() - np.abs(dense)), -2e-12)

    def test_zero_envelopes_and_zero_repair(self):
        zero = TensorTrain.zeros(self.operator.shape)
        for envelope in (
            quadratic_envelope(zero, self.operator),
            core_envelope(zero),
            controlled_rank_one_envelope(
                zero, self.operator, 1, 2.0, structure_certified=True
            ),
        ):
            self.assertEqual(envelope.norm(), 0.0)
        result = repair_feasible(zero, zero, self.operator)
        self.assertEqual(result.error_bound, 0.0)
        self.assertEqual(result.tensor.norm(), 0.0)

    def test_repair_dominates_signed_original_and_energy_bound(self):
        original = TensorTrain.from_dense(self.rng.normal(size=self.operator.shape))
        candidate = TensorTrain.from_dense(self.rng.normal(size=self.operator.shape))
        for method in ("quadratic", "core"):
            result = repair_feasible(original, candidate, self.operator, method=method)
            self.assertGreaterEqual(
                np.min(result.tensor.to_dense() - original.to_dense()), -2e-11
            )
            actual = self.operator.energy_norm(result.tensor.subtract(original))
            self.assertLessEqual(actual, result.error_bound + 1e-11)
            self.assertAlmostEqual(
                result.error_bound, result.raw_error + result.correction_energy
            )
            self.assertEqual(result.result_storage, result.tensor.storage_size)

    def _additive_residual(self):
        residual = TensorTrain.zeros(self.operator.shape)
        kappa = 1.0
        for k, (b, mass, vectors) in enumerate(
            zip(self.operator.ground_factors, self.operator.mass, self.operator.eigenvectors)
        ):
            z = vectors[:, 1]
            factors = list(self.operator.ground_factors)
            factors[k] = z
            residual = residual.add(TensorTrain.rank_one(factors).scaled(0.4 + k))
            mu = b * (mass @ b)
            ratio = z / b
            local = np.max(np.abs(ratio)) / np.sqrt(mu @ (ratio * ratio))
            kappa = max(kappa, float(local))
        return residual, kappa

    def test_controlled_rank_one_with_declared_local_spaces(self):
        residual, kappa = self._additive_residual()
        envelope = controlled_rank_one_envelope(
            residual, self.operator, 1, kappa, structure_certified=True
        )
        self.assertTrue(all(rank == 1 for rank in envelope.ranks))
        self.assertGreaterEqual(
            np.min(envelope.to_dense() - np.abs(residual.to_dense())), -2e-11
        )

    def test_structural_assertions_and_invalid_inputs(self):
        residual, kappa = self._additive_residual()
        with self.assertRaises(ValueError):
            controlled_rank_one_envelope(residual, self.operator, 1, kappa)
        for degree, local_bound in [(-1, 2.0), (True, 2.0), (1.5, 2.0), (1, 0.5), (1, np.inf)]:
            with self.subTest(degree=degree, kappa=local_bound):
                with self.assertRaises(ValueError):
                    controlled_rank_one_envelope(
                        residual, self.operator, degree, local_bound, structure_certified=True
                    )
        with self.assertRaises(ValueError):
            controlled_rank_one_envelope(
                residual, self.operator, 0, kappa, structure_certified=True
            )
        with self.assertRaises(ValueError):
            repair_feasible(residual, residual, self.operator, method="unknown")
        with self.assertRaises(ValueError):
            repair_feasible(residual, residual, self.operator, method="controlled_rank_one")

    def test_repairs_do_not_materialize_dense_tensor(self):
        residual, kappa = self._additive_residual()
        with patch.object(TensorTrain, "to_dense", side_effect=AssertionError("dense call")):
            quadratic_envelope(residual, self.operator)
            core_envelope(residual)
            controlled_rank_one_envelope(
                residual, self.operator, 1, kappa, structure_certified=True
            )
            repair_feasible(residual, residual.scaled(0.8), self.operator)


if __name__ == "__main__":
    unittest.main()
