"""Independent dense and core-only checks for the tensor-train implementation."""
from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import numpy as np

from tt_obstacle.tensor import TensorTrain


class TensorTrainTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20310920)

    def random_tt(self, shape=(3, 4, 2, 3), ranks=(2, 3, 2)):
        ranks = (1,)+tuple(ranks)+(1,)
        return TensorTrain([self.rng.normal(size=(ranks[k], n, ranks[k+1]))
                            for k, n in enumerate(shape)])

    def test_constructor_owns_read_only_cores(self):
        source = [np.arange(6.).reshape(1, 3, 2), np.ones((2, 4, 1))]
        tensor = TensorTrain(source)
        before = tensor.to_dense()
        source[0].fill(0.)
        np.testing.assert_array_equal(tensor.to_dense(), before)
        self.assertEqual(tensor.shape, (3, 4))
        self.assertEqual(tensor.ranks, (2,))
        self.assertEqual(tensor.storage_size, 14)
        with self.assertRaises(ValueError):
            tensor.cores[0][0, 0, 0] = 99.
        copied = tensor.copy()
        for original, duplicate in zip(tensor.cores, copied.cores):
            self.assertFalse(np.shares_memory(original, duplicate))

    def test_invalid_cores_are_rejected(self):
        invalid = [[], [np.ones((3, 2))], [np.ones((2, 2, 1))],
                   [np.ones((1, 2, 2))], [np.ones((1, 0, 1))],
                   [np.ones((1, 2, 0)), np.ones((0, 2, 1))],
                   [np.ones((1, 2, 2)), np.ones((3, 2, 1))],
                   [np.array([[[np.nan]]])], [np.array([[[np.inf]]])],
                   [np.array([[[1.+0.j]]])], [np.array([[['1']]])]]
        for cores in invalid:
            with self.subTest(cores=repr(cores)), self.assertRaises(ValueError):
                TensorTrain(cores)

    def test_rank_one_zeros_and_entries(self):
        factors = [np.array([1., -2.]), np.array([3., 0., 4.]), np.array([-1., 2.])]
        tensor = TensorTrain.rank_one(factors)
        dense = np.einsum("i,j,k->ijk", *factors)
        np.testing.assert_allclose(tensor.to_dense(), dense)
        self.assertEqual(tensor.ranks, (1, 1))
        self.assertAlmostEqual(tensor.evaluate((1, 2, 1)), dense[1, 2, 1])
        zeros = TensorTrain.zeros((2, 3, 4))
        np.testing.assert_array_equal(zeros.to_dense(), np.zeros((2, 3, 4)))
        self.assertEqual(zeros.norm(), 0.)
        self.assertEqual(zeros.round(rtol=0.1).ranks, (1, 1))
        self.assertEqual(TensorTrain.from_dense(np.zeros((2, 3, 4))).ranks, (1, 1))
        with self.assertRaises(ValueError):
            tensor.evaluate((0, 0))
        with self.assertRaises(ValueError):
            tensor.evaluate((0, 1.5, 0))
        with self.assertRaises(IndexError):
            tensor.evaluate((0, -1, 0))
        with self.assertRaises(IndexError):
            tensor.evaluate((0, 3, 0))
        with self.assertRaises(ValueError):
            tensor.to_dense(max_entries=11)

    def test_exact_dense_svd_and_norm(self):
        for shape in [(7,), (3, 5), (2, 3, 4, 2)]:
            with self.subTest(shape=shape):
                dense = self.rng.normal(size=shape)
                tensor = TensorTrain.from_dense(dense)
                np.testing.assert_allclose(tensor.to_dense(), dense, rtol=2e-13, atol=2e-13)
                self.assertAlmostEqual(tensor.norm(), np.linalg.norm(dense), places=11)
                np.testing.assert_allclose(tensor.round().to_dense(), dense, atol=3e-13)

    def test_arithmetic_and_nonmutation(self):
        first, second = self.random_tt(), self.random_tt(ranks=(3, 2, 2))
        a, b = first.to_dense(), second.to_dense()
        snapshots = [[core.copy() for core in tt.cores] for tt in (first, second)]
        checks = [(first.add(second), a+b), (first.subtract(second), a-b),
                  (first.hadamard(second), a*b), (first.scaled(-2.5), -2.5*a)]
        for actual, expected in checks:
            np.testing.assert_allclose(actual.to_dense(), expected, rtol=3e-13, atol=3e-13)
        for tensor, previous in zip((first, second), snapshots):
            for actual, expected in zip(tensor.cores, previous):
                np.testing.assert_array_equal(actual, expected)
        self.assertEqual(first.hadamard(second).ranks,
                         tuple(a*b for a, b in zip(first.ranks, second.ranks)))
        cancelled = first.subtract(first)
        self.assertLess(cancelled.norm(), 1e-12*first.norm())

    def test_symmetric_square_values_and_rank_bound(self):
        for shape, ranks in [((5,), ()), ((3, 4, 2, 3), (2, 3, 2)),
                             ((2, 3, 2), (4, 5))]:
            with self.subTest(shape=shape, ranks=ranks):
                tensor = self.random_tt(shape, ranks)
                dense = tensor.to_dense()
                squared = tensor.square()
                np.testing.assert_allclose(squared.to_dense(), dense*dense,
                                           rtol=4e-13, atol=4e-12)
                self.assertEqual(squared.ranks, tuple(r*(r+1)//2 for r in ranks))
                if any(r > 1 for r in ranks):
                    self.assertLess(squared.storage_size, tensor.hadamard(tensor).storage_size)

    def test_local_maps_against_independent_dense_einsum(self):
        tensor = self.random_tt((2, 3, 4), (2, 3))
        dense = tensor.to_dense()
        matrices = [self.rng.normal(size=(3, 2)), self.rng.normal(size=(2, 3)),
                    self.rng.normal(size=(5, 4))]
        expected = np.einsum("ai,bj,ck,ijk->abc", *matrices, dense)
        np.testing.assert_allclose(tensor.mode_product(matrices).to_dense(), expected,
                                   rtol=5e-13, atol=5e-13)
        factors = [self.rng.normal(size=n) for n in tensor.shape]
        expected_scaled = dense*np.einsum("i,j,k->ijk", *factors)
        np.testing.assert_allclose(tensor.scale_modes(factors).to_dense(), expected_scaled,
                                   rtol=3e-13, atol=3e-13)
        np.testing.assert_array_equal(tensor.mode_product([None]*3).to_dense(), dense)
        self.assertEqual(tensor.mode_product(matrices).ranks, tensor.ranks)

    def test_rounding_global_error_and_actual_rank_reduction(self):
        low_rank = self.random_tt((4, 3, 5, 3), (2, 2, 2)).to_dense()
        dense = low_rank+1e-4*self.rng.normal(size=low_rank.shape)
        original = TensorTrain.from_dense(dense)
        snapshots = [core.copy() for core in original.cores]
        rounded = original.round(rtol=1e-3)
        self.assertLessEqual(np.linalg.norm(rounded.to_dense()-dense),
                             1.00001e-3*np.linalg.norm(dense))
        self.assertTrue(all(rank <= 2 for rank in rounded.ranks))
        self.assertLess(rounded.storage_size, original.storage_size)
        for actual, expected in zip(original.cores, snapshots):
            np.testing.assert_array_equal(actual, expected)
        compressed_directly = TensorTrain.from_dense(dense, rtol=1e-3)
        self.assertLessEqual(np.linalg.norm(compressed_directly.to_dense()-dense),
                             1.00001e-3*np.linalg.norm(dense))

    def test_rank_caps_can_override_error_tolerance(self):
        dense = np.diag([3., 1.])
        original = TensorTrain.from_dense(dense)
        for result in (original.round(rtol=.01, max_rank=1),
                       TensorTrain.from_dense(dense, rtol=.01, max_rank=1)):
            self.assertEqual(result.ranks, (1,))
            self.assertAlmostEqual(np.linalg.norm(result.to_dense()-dense), 1.)
        general = self.random_tt()
        capped = general.round(max_rank=(1, 2, 1))
        self.assertTrue(all(actual <= cap for actual, cap in zip(capped.ranks, (1, 2, 1))))

    def test_single_mode_operations(self):
        tensor = TensorTrain.from_dense(np.array([1., -2., 3.]))
        self.assertEqual(tensor.ranks, ())
        np.testing.assert_allclose(tensor.add(tensor).to_dense(), [2., -4., 6.])
        np.testing.assert_allclose(tensor.square().to_dense(), [1., 4., 9.])
        np.testing.assert_allclose(tensor.round(rtol=.5, max_rank=1).to_dense(), [1., -2., 3.])
        np.testing.assert_allclose(tensor.round(max_rank=[]).to_dense(), [1., -2., 3.])
        self.assertEqual(tensor.evaluate([1]), -2.)

    def test_representable_large_norm_and_square(self):
        tensor = TensorTrain.rank_one([np.array([1e154])])
        with np.errstate(over="raise", invalid="raise"):
            self.assertEqual(tensor.norm(), 1e154)
            self.assertEqual(tensor.square().evaluate([0]), 1e308)

    def test_high_dimension_operations_do_not_densify(self):
        d = 64
        factor = np.array([1., .5])
        tensor = TensorTrain.rank_one([factor]*d)
        with self.assertRaises(ValueError):
            tensor.to_dense()
        with patch.object(TensorTrain, "to_dense", side_effect=AssertionError("densified")), \
             patch.object(TensorTrain, "from_dense", side_effect=AssertionError("densified")):
            self.assertEqual(tensor.storage_size, 2*d)
            self.assertAlmostEqual(tensor.evaluate([1]*d), .5**d)
            self.assertAlmostEqual(tensor.norm()/(1.25**(d/2)), 1., places=12)
            combined = tensor.add(tensor.scaled(2.)).round(rtol=1e-12)
            self.assertTrue(all(rank == 1 for rank in combined.ranks))
            self.assertAlmostEqual(combined.evaluate([0]*d), 3., places=10)
            squared = tensor.square().round(rtol=1e-12)
            self.assertAlmostEqual(squared.evaluate([0]*d), 1., places=11)
            product = tensor.hadamard(tensor)
            self.assertAlmostEqual(product.evaluate([1]*d), .25**d)
            mapped = tensor.mode_product([np.eye(2)]*d).scale_modes([np.ones(2)]*d)
            self.assertAlmostEqual(mapped.evaluate([0]*d), 1.)

    def test_invalid_operation_arguments(self):
        tensor = TensorTrain.rank_one([np.ones(2), np.ones(3)])
        for tolerance in [-1., np.nan, np.inf, 1j, True, "0.1"]:
            with self.subTest(rtol=tolerance), self.assertRaises(ValueError):
                tensor.round(rtol=tolerance)
        for cap in [0, -1, 1.5, True, [], [1, 2], [0], [1.5]]:
            with self.subTest(max_rank=cap), self.assertRaises(ValueError):
                tensor.round(max_rank=cap)
        for scalar in [np.nan, np.inf, 1j, True]:
            with self.subTest(scalar=scalar), self.assertRaises(ValueError):
                tensor.scaled(scalar)
        for dense in [np.array(1.), np.ones((2, 0)), np.array([1j]), np.array([np.inf])]:
            with self.subTest(dense=repr(dense)), self.assertRaises(ValueError):
                TensorTrain.from_dense(dense)
        with self.assertRaises(ValueError):
            tensor.add(TensorTrain.zeros((2, 2)))
        with self.assertRaises(TypeError):
            tensor.add(np.ones((2, 3)))
        with self.assertRaises(ValueError):
            tensor.mode_product([np.eye(2)])
        with self.assertRaises(ValueError):
            tensor.mode_product([np.eye(3), np.eye(3)])
        with self.assertRaises(ValueError):
            tensor.scale_modes([np.ones(2), np.ones(2)])
        for shape in [(), (0,), (1.5,), (True,)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                TensorTrain.zeros(shape)


if __name__ == "__main__":
    unittest.main()
