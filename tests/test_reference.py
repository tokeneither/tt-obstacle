import unittest

import numpy as np

from tt_obstacle.operators import Q1Operator
from tt_obstacle.reference import solve_obstacle


class ReferenceTests(unittest.TestCase):
    def test_scalar_contact_and_free(self):
        contact = solve_obstacle([[2.0]], [-3.0], [0.5])
        np.testing.assert_array_equal(contact.solution, [0.5])
        np.testing.assert_array_equal(contact.multiplier, [4.0])
        free = solve_obstacle([[2.0]], [3.0], [0.5])
        np.testing.assert_array_equal(free.solution, [1.5])
        np.testing.assert_array_equal(free.multiplier, [0.0])

    def test_prescribed_kkt_with_nonzero_obstacle_and_full_q1(self):
        operator = Q1Operator.uniform((2, 3))
        stiffness = operator.to_dense()
        obstacle = np.array([-0.3, 0.2, 0.4, -0.7, 0.8, 0.1])
        slack = np.array([0., 0.5, 0., 0.25, 0., 0.9])
        multiplier = np.array([0.7, 0., 1.3, 0., 0.6, 0.])
        exact = obstacle + slack
        load = stiffness @ exact - multiplier
        result = solve_obstacle(stiffness, load, obstacle)
        np.testing.assert_allclose(result.solution, exact, atol=2e-14)
        np.testing.assert_allclose(result.multiplier, multiplier, atol=2e-14)
        np.testing.assert_allclose(result.multiplier,
                                   stiffness @ result.solution - load, atol=1e-15)
        self.assertFalse(result.solution.flags.writeable)

    def test_random_spd_prescribed_contact_patterns(self):
        rng = np.random.default_rng(447)
        for n in (2, 5, 8):
            for _ in range(3):
                factor = rng.normal(size=(n, n))
                stiffness = factor.T @ factor + np.eye(n)
                obstacle = rng.normal(size=n)
                active = rng.random(n) < 0.5
                slack = np.where(active, 0, rng.uniform(0.1, 1, n))
                multiplier = np.where(active, rng.uniform(0.1, 1, n), 0)
                exact = obstacle + slack
                result = solve_obstacle(stiffness, stiffness @ exact - multiplier, obstacle)
                np.testing.assert_allclose(result.solution, exact, atol=2e-13)
                np.testing.assert_allclose(result.multiplier, multiplier, atol=2e-13)

    def test_tolerance_does_not_clip_returned_raw_candidate(self):
        result = solve_obstacle([[1.0]], [-1e-12], [0.0], tol=1e-10)
        self.assertEqual(result.solution[0], -1e-12)
        self.assertEqual(result.multiplier[0], 0.0)

    def test_guard_and_validation(self):
        with self.assertRaises(ValueError):
            solve_obstacle(np.eye(17), np.zeros(17), np.zeros(17))
        for bad in (np.empty((0, 0)), [[1., 2.], [0., 1.]], [[-1.]]):
            n = len(bad)
            with self.assertRaises(ValueError):
                solve_obstacle(bad, np.zeros(n), np.zeros(n))
        with self.assertRaises(ValueError):
            solve_obstacle(np.eye(2), np.zeros(3), np.zeros(2))
        with self.assertRaises(ValueError):
            solve_obstacle([[1]], [np.nan], [0])
        with self.assertRaises(ValueError):
            solve_obstacle([[1]], [0], [0], tol=-1)
        with self.assertRaises(ValueError):
            solve_obstacle([[1]], [0], [0], max_unknowns=True)


if __name__ == "__main__":
    unittest.main()

