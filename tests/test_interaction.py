import itertools
import unittest

import numpy as np

from tt_obstacle.tensor import TensorTrain
from tt_obstacle.operators import Q1Operator
from tt_obstacle.interaction import (project_degree, project_coordinates,
                                    project_spectral, spectral_feature_bound,
                                    interaction_amplification_bound)


def mode(x, matrix, axis):
    return np.moveaxis(np.tensordot(matrix, x, axes=(1, axis)), 0, axis)


class InteractionTests(unittest.TestCase):
    def test_mpo_matches_independent_subset_expansion(self):
        rng = np.random.default_rng(24)
        op = Q1Operator.uniform((3, 2, 3, 2))
        x = rng.normal(size=op.shape)
        tt = TensorTrain.from_dense(x)
        for p in range(5):
            expected = np.zeros_like(x)
            for mask in itertools.product((0, 1), repeat=4):
                if sum(mask) > p:
                    continue
                term = x.copy()
                for k, (active, proj) in enumerate(zip(mask, op.projectors)):
                    term = mode(term, np.eye(len(proj)) - proj if active else proj, k)
                expected += term
            actual = project_degree(tt, op.projectors, p)
            np.testing.assert_allclose(actual.to_dense(), expected, atol=2e-12)
            self.assertTrue(all(a <= (p + 1) * b for a, b in zip(actual.ranks, tt.ranks)))
            again = project_degree(actual, op.projectors, p)
            np.testing.assert_allclose(again.to_dense(), expected, atol=5e-12)
            residual = tt.subtract(actual)
            self.assertLess(abs(op.energy_inner(residual, actual)), 1e-9)

    def test_one_mode_and_positive_coordinate_projection(self):
        op = Q1Operator.uniform((3,))
        x = TensorTrain.from_dense(np.array([1., -2., 3.]))
        np.testing.assert_allclose(project_degree(x, op.projectors, 0).to_dense(),
                                   op.projectors[0] @ x.to_dense(), atol=1e-13)
        op = Q1Operator.uniform((3, 3, 2))
        x = TensorTrain.from_dense(np.arange(18.).reshape(op.shape))
        y = project_coordinates(x, op.projectors, [1])
        self.assertGreaterEqual(y.to_dense().min(), -1e-13)
        np.testing.assert_allclose(project_degree(y, op.projectors, 1).to_dense(),
                                   y.to_dense(), atol=1e-11)

    def test_spectral_projection_and_whole_space_bound(self):
        op = Q1Operator.uniform((5, 4))
        x = TensorTrain.rank_one([v[:, -1] for v in op.eigenvectors])
        projected = project_spectral(x, op, 2)
        self.assertLess(op.mass_norm(projected), 1e-12)
        kappa = spectral_feature_bound(op, 3)
        for b, m, vectors in zip(op.ground_factors, op.mass, op.eigenvectors):
            z = (vectors[:, 1] + .7 * vectors[:, 2]) / b
            self.assertLessEqual(np.max(np.abs(z)),
                                 kappa * np.sqrt(np.sum(b * (m @ b) * z * z)) + 1e-12)
        self.assertGreater(interaction_amplification_bound(1, kappa, op.comparison_constant), 1)

    def test_high_dimension_without_dense_arrays(self):
        d = 40
        op = Q1Operator.uniform((2,) * d)
        tt = TensorTrain.rank_one([np.array([1., 1.1])] * d)
        p = project_degree(tt, op.projectors, 2)
        self.assertLessEqual(max(p.ranks), 3)
        self.assertTrue(np.isfinite(p.evaluate([0] * d)))

    def test_validation(self):
        tt = TensorTrain.rank_one([np.ones(2), np.ones(2)])
        ps = [np.eye(2)] * 2
        for p in (-1, .5, True):
            with self.assertRaises(ValueError):
                project_degree(tt, ps, p)
        with self.assertRaises(ValueError):
            project_degree(tt, [2 * np.eye(2)] * 2, 1)
        with self.assertRaises(ValueError):
            project_degree(tt, [np.array([[1, 1j], [0, 0]])] * 2, 1)
        for keep in ([2], [-1], [0, 0], [True]):
            with self.assertRaises(ValueError):
                project_coordinates(tt, ps, keep)


if __name__ == "__main__":
    unittest.main()
