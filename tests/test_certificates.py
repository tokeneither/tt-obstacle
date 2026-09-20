import math
import unittest

import numpy as np

from tt_obstacle.certificates import energy_error_bound
from tt_obstacle.operators import Q1Operator
from tt_obstacle.reference import solve_obstacle


class CertificateTests(unittest.TestCase):
    def test_scalar_sharpness_for_residual_and_complementarity(self):
        for r, c in ((0., 0.), (0., 2.), (3., 0.), (1.7, 0.6)):
            exact_bound = r / 2 + math.hypot(r / 2, math.sqrt(c))
            multiplier = exact_bound - r
            result = energy_error_bound([[1.]], [0.], [0.],
                                        [exact_bound], [multiplier])
            self.assertAlmostEqual(result.residual_norm, r, places=14)
            self.assertAlmostEqual(result.complementarity, c, places=14)
            self.assertAlmostEqual(result.sharp_bound, exact_bound, places=14)
            self.assertLessEqual(result.sharp_bound, result.gap_bound + 1e-14)

    def test_actual_error_is_bounded_for_random_feasible_pairs(self):
        rng = np.random.default_rng(333)
        stiffness = Q1Operator.uniform((2, 3)).to_dense()
        obstacle = rng.normal(size=6)
        active = np.array([True, False, True, False, True, False])
        exact = obstacle + np.where(active, 0., rng.uniform(0.2, 1., 6))
        exact_multiplier = np.where(active, rng.uniform(0.3, 1., 6), 0.)
        load = stiffness @ exact - exact_multiplier
        reference = solve_obstacle(stiffness, load, obstacle)
        np.testing.assert_allclose(reference.solution, exact, atol=1e-13)
        for _ in range(12):
            candidate = obstacle + rng.uniform(0., 1.2, 6)
            multiplier = rng.uniform(0., 1., 6)
            result = energy_error_bound(stiffness, load, obstacle, candidate, multiplier)
            error = candidate - exact
            actual_error = math.sqrt(error @ stiffness @ error)
            self.assertLessEqual(actual_error, result.sharp_bound + 1e-12)
            self.assertLessEqual(result.sharp_bound, result.gap_bound + 1e-12)
            residual = stiffness @ candidate - load - multiplier
            expected_r = math.sqrt(residual @ np.linalg.solve(stiffness, residual))
            self.assertAlmostEqual(result.residual_norm, expected_r, places=12)

    def test_zero_residual_does_not_hide_complementarity_error(self):
        result = energy_error_bound([[1.]], [0.], [0.], [2.], [2.])
        self.assertEqual(result.residual_norm, 0)
        self.assertEqual(result.complementarity, 4)
        self.assertAlmostEqual(result.sharp_bound, 2, places=14)

    def test_strict_feasibility_even_inside_tolerance(self):
        with self.assertRaisesRegex(ValueError, "candidate"):
            energy_error_bound([[1]], [0], [0], [-1e-16], [0], tol=1e-6)
        with self.assertRaisesRegex(ValueError, "multiplier"):
            energy_error_bound([[1]], [0], [0], [0], [-1e-16], tol=1e-6)
        with self.assertRaises(ValueError):
            energy_error_bound([[1]], [0], [0], [np.nan], [0])
        with self.assertRaises(ValueError):
            energy_error_bound([[-1]], [0], [0], [0], [0])
        with self.assertRaises(ValueError):
            energy_error_bound([[1]], [0, 0], [0], [0], [0])

    def test_representable_extreme_residuals(self):
        for value in (1e-200, 1e200):
            result = energy_error_bound([[1.]], [0.], [0.], [value], [0.])
            self.assertTrue(math.isfinite(result.sharp_bound))
            self.assertAlmostEqual(result.sharp_bound / value, 1.0, places=14)

    def test_tiny_complementarity_does_not_produce_false_zero_bound(self):
        value = 1e-200
        result = energy_error_bound([[1.]], [0.], [0.], [value], [value])
        self.assertEqual(result.residual_norm, 0)
        self.assertAlmostEqual(result.sharp_bound / value, 1.0, places=14)


if __name__ == "__main__":
    unittest.main()
