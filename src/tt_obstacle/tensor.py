"""Real tensor trains with core-only arithmetic and Frobenius rounding.

Dense conversion is explicit and bounded. All operations return new tensor
trains; the constructor owns copies of its read-only, float64 cores.
"""
from __future__ import annotations

import math
import numbers
from typing import Sequence

import numpy as np


def _real_array(value, *, ndim: int, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "biuf" or np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain real numeric values")
    if raw.ndim != ndim or any(size == 0 for size in raw.shape):
        raise ValueError(f"{name} must be a nonempty {ndim}-dimensional array")
    array = np.array(raw, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _positive_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be a positive integer")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _real_scalar(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite real scalar")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be a finite real scalar")
    return scalar


def _options(order: int, rtol, max_rank):
    tolerance = _real_scalar(rtol, "rtol")
    if tolerance < 0:
        raise ValueError("rtol must be nonnegative")
    if max_rank is None:
        caps = (None,)*(order-1)
    elif isinstance(max_rank, numbers.Integral):
        cap = _positive_integer(max_rank, "max_rank")
        caps = (cap,)*(order-1)
    else:
        try:
            values = tuple(max_rank)
        except TypeError as exc:
            raise ValueError("max_rank must be an integer or one cap per internal bond") from exc
        if len(values) != order-1:
            raise ValueError("max_rank must have one cap per internal bond")
        caps = tuple(_positive_integer(value, "max_rank") for value in values)
    return tolerance, caps


def _frobenius(array: np.ndarray) -> float:
    """Avoid overflow from squaring representable large entries."""
    scale = float(np.max(np.abs(array)))
    return 0. if scale == 0. else scale*float(np.linalg.norm(array/scale))


def _left_orthogonalize(cores):
    result = [core.copy() for core in cores]
    for axis in range(len(result)-1):
        left, size, right = result[axis].shape
        q, r = np.linalg.qr(result[axis].reshape(left*size, right), mode="reduced")
        result[axis] = q.reshape(left, size, q.shape[1])
        result[axis+1] = np.tensordot(r, result[axis+1], axes=(1, 0))
    return result


def _retained_rank(singular_values, budget: float, cap) -> int:
    """Smallest positive rank meeting a local absolute error budget."""
    scale = float(singular_values[0])
    if scale == 0:
        return 1
    relative_budget = budget/scale
    limit = relative_budget*relative_budget
    discarded = 0.
    rank = len(singular_values)
    while rank > 1:
        candidate = discarded+float(singular_values[rank-1]/scale)**2
        if candidate > limit:
            break
        discarded = candidate
        rank -= 1
    if cap is not None:
        rank = min(rank, cap)
    return rank


class TensorTrain:
    """A real TT with cores of shape ``(r_left, mode_size, r_right)``.

    Adjacent ranks must agree and both exterior ranks must be one. Zero
    tensors retain positive core dimensions; a rank of zero is never used.
    Inputs are copied, and public cores are read-only arrays.
    """

    __slots__ = ("_cores",)

    def __init__(self, cores: Sequence[np.ndarray]):
        try:
            values = tuple(cores)
        except TypeError as exc:
            raise ValueError("cores must be a nonempty sequence") from exc
        if not values:
            raise ValueError("cores must be a nonempty sequence")
        copied = tuple(_real_array(core, ndim=3, name=f"core {axis}")
                       for axis, core in enumerate(values))
        if copied[0].shape[0] != 1 or copied[-1].shape[2] != 1:
            raise ValueError("both exterior TT ranks must equal one")
        for left, right in zip(copied, copied[1:]):
            if left.shape[2] != right.shape[0]:
                raise ValueError("adjacent TT ranks must agree")
        for core in copied:
            core.flags.writeable = False
        self._cores = copied

    @property
    def cores(self) -> tuple[np.ndarray, ...]:
        return self._cores

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(core.shape[1] for core in self._cores)

    @property
    def ranks(self) -> tuple[int, ...]:
        """The internal ranks, with no exterior ones included."""
        return tuple(core.shape[2] for core in self._cores[:-1])

    @property
    def storage_size(self) -> int:
        """Number of scalar entries stored in all cores."""
        return sum(core.size for core in self._cores)

    @classmethod
    def rank_one(cls, factors):
        factors = tuple(factors)
        if not factors:
            raise ValueError("factors must be a nonempty sequence")
        return cls([_real_array(factor, ndim=1, name=f"factor {axis}").reshape(1, -1, 1)
                    for axis, factor in enumerate(factors)])

    @classmethod
    def zeros(cls, shape):
        shape = tuple(_positive_integer(size, "mode size") for size in shape)
        if not shape:
            raise ValueError("shape must contain at least one mode")
        factors = [np.ones(size) for size in shape]
        factors[0].fill(0.)
        return cls.rank_one(factors)

    @classmethod
    def from_dense(cls, array, *, rtol=0.0, max_rank=None):
        """TT-SVD of a dense array.

        Without a binding ``max_rank``, the Frobenius error is at most
        ``rtol * ||array||_F``, up to floating-point roundoff. A rank cap takes
        precedence and may violate that tolerance. ``max_rank`` is either a
        positive integer or a sequence of positive caps, one per TT bond.
        """
        raw = np.asarray(array)
        value = _real_array(raw, ndim=raw.ndim, name="array")
        if value.ndim == 0:
            raise ValueError("array must have at least one mode")
        tolerance, caps = _options(value.ndim, rtol, max_rank)
        if value.ndim == 1:
            return cls([value.reshape(1, -1, 1)])
        budget = tolerance*_frobenius(value)/math.sqrt(value.ndim-1)
        remainder = value
        cores = []
        previous = 1
        for axis, size in enumerate(value.shape[:-1]):
            matrix = remainder.reshape(previous*size, -1)
            u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
            rank = _retained_rank(singular, budget, caps[axis])
            cores.append(u[:, :rank].reshape(previous, size, rank))
            remainder = singular[:rank, None]*vh[:rank]
            previous = rank
        cores.append(remainder.reshape(previous, value.shape[-1], 1))
        return cls(cores)

    def to_dense(self, max_entries=1_000_000) -> np.ndarray:
        """Materialize explicitly, rejecting arrays above ``max_entries``."""
        limit = _positive_integer(max_entries, "max_entries")
        if math.prod(self.shape) > limit:
            raise ValueError("dense tensor exceeds max_entries")
        value = self._cores[0][0]
        for core in self._cores[1:]:
            value = np.tensordot(value, core, axes=(-1, 0))
        return value[..., 0].copy()

    def evaluate(self, indices) -> float:
        """Evaluate one entry by multiplying core slices."""
        indices = tuple(indices)
        if len(indices) != len(self._cores):
            raise ValueError("provide one index per mode")
        value = np.ones(1)
        for axis, (index, core) in enumerate(zip(indices, self._cores)):
            if isinstance(index, (bool, np.bool_)) or not isinstance(index, numbers.Integral):
                raise ValueError("indices must be integers")
            if index < 0 or index >= core.shape[1]:
                raise IndexError(f"index out of range in mode {axis}")
            value = value @ core[:, int(index), :]
        return float(value[0])

    def norm(self) -> float:
        """Frobenius norm from left QR orthogonalization, without densifying."""
        return _frobenius(_left_orthogonalize(self._cores)[-1])

    def copy(self):
        return type(self)(self._cores)

    def scaled(self, scalar):
        scalar = _real_scalar(scalar, "scalar")
        cores = list(self._cores)
        cores[0] = scalar*cores[0]
        return type(self)(cores)

    def _require_matching_shape(self, other):
        if not isinstance(other, TensorTrain):
            raise TypeError("operand must be a TensorTrain")
        if self.shape != other.shape:
            raise ValueError("tensor shapes must agree")

    def add(self, other):
        self._require_matching_shape(other)
        if len(self._cores) == 1:
            return type(self)([self._cores[0]+other._cores[0]])
        cores = [np.concatenate((self._cores[0], other._cores[0]), axis=2)]
        for first, second in zip(self._cores[1:-1], other._cores[1:-1]):
            core = np.zeros((first.shape[0]+second.shape[0], first.shape[1],
                             first.shape[2]+second.shape[2]))
            core[:first.shape[0], :, :first.shape[2]] = first
            core[first.shape[0]:, :, first.shape[2]:] = second
            cores.append(core)
        cores.append(np.concatenate((self._cores[-1], other._cores[-1]), axis=0))
        return type(self)(cores)

    def subtract(self, other):
        self._require_matching_shape(other)
        return self.add(other.scaled(-1.))

    def hadamard(self, other):
        """Entrywise product with product internal ranks."""
        self._require_matching_shape(other)
        cores = []
        for first, second in zip(self._cores, other._cores):
            product = np.einsum("aib,cid->acibd", first, second, optimize=True)
            cores.append(product.reshape(first.shape[0]*second.shape[0], first.shape[1],
                                         first.shape[2]*second.shape[2]))
        return type(self)(cores)

    def square(self):
        """Entrywise square using orthonormal symmetric-square bond bases.

        Each internal rank ``r`` becomes ``r*(r+1)//2``, rather than ``r*r``.
        The basis consists of diagonal tensors and normalized symmetric pairs;
        no dense physical tensor or full Kronecker-product core is formed.
        """
        cores = []
        for core in self._cores:
            left_a, left_b = np.triu_indices(core.shape[0])
            right_a, right_b = np.triu_indices(core.shape[2])
            left_diagonal = (left_a == left_b)[:, None]
            right_diagonal = (right_a == right_b)[None, :]
            mixed = np.logical_xor(left_diagonal, right_diagonal)
            both_off_diagonal = ~left_diagonal & ~right_diagonal
            squared = np.empty((len(left_a), core.shape[1], len(right_a)))
            for index in range(core.shape[1]):
                matrix = core[:, index, :]
                first = matrix[left_a[:, None], right_a[None, :]]
                first = first*matrix[left_b[:, None], right_b[None, :]]
                second = matrix[left_a[:, None], right_b[None, :]]
                second = second*matrix[left_b[:, None], right_a[None, :]]
                # Separate diagonal cases to avoid unnecessary doubling and
                # division, which can overflow or underflow a finite square.
                first[mixed] *= math.sqrt(2.)
                first[both_off_diagonal] += second[both_off_diagonal]
                squared[:, index, :] = first
            cores.append(squared)
        return type(self)(cores)

    def mode_product(self, matrices):
        """Apply local matrices shaped ``(new_size, old_size)``; None is identity."""
        matrices = tuple(matrices)
        if len(matrices) != len(self._cores):
            raise ValueError("provide one matrix per mode")
        cores = []
        for axis, (matrix, core) in enumerate(zip(matrices, self._cores)):
            if matrix is None:
                cores.append(core)
                continue
            matrix = _real_array(matrix, ndim=2, name=f"mode matrix {axis}")
            if matrix.shape[1] != core.shape[1]:
                raise ValueError("mode matrix has an incompatible input size")
            cores.append(np.einsum("ji,aib->ajb", matrix, core, optimize=True))
        return type(self)(cores)

    def scale_modes(self, factors):
        """Multiply entries by the separable tensor formed from ``factors``."""
        factors = tuple(factors)
        if len(factors) != len(self._cores):
            raise ValueError("provide one scaling vector per mode")
        cores = []
        for axis, (factor, core) in enumerate(zip(factors, self._cores)):
            factor = _real_array(factor, ndim=1, name=f"mode scaling {axis}")
            if len(factor) != core.shape[1]:
                raise ValueError("mode scaling has an incompatible length")
            cores.append(core*factor[None, :, None])
        return type(self)(cores)

    def round(self, *, rtol=0.0, max_rank=None):
        """QR/SVD rounding on TT cores only.

        With no binding rank cap, ``||self-result||_F <= rtol*||self||_F``
        up to floating-point roundoff. Every bond receives absolute budget
        ``rtol*||self||_F/sqrt(d-1)``. A positive scalar ``max_rank`` or one cap
        per internal bond takes precedence over that error guarantee.
        """
        tolerance, caps = _options(len(self._cores), rtol, max_rank)
        cores = _left_orthogonalize(self._cores)
        if len(cores) == 1:
            return type(self)(cores)
        budget = tolerance*_frobenius(cores[-1])/math.sqrt(len(cores)-1)
        for axis in range(len(cores)-1, 0, -1):
            left, size, right = cores[axis].shape
            u, singular, vh = np.linalg.svd(cores[axis].reshape(left, size*right),
                                           full_matrices=False)
            rank = _retained_rank(singular, budget, caps[axis-1])
            cores[axis] = vh[:rank].reshape(rank, size, right)
            transfer = u[:, :rank]*singular[:rank]
            cores[axis-1] = np.tensordot(cores[axis-1], transfer, axes=(2, 0))
        return type(self)(cores)

    def __add__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.subtract(other)

    def __repr__(self):
        return f"TensorTrain(shape={self.shape}, ranks={self.ranks}, storage_size={self.storage_size})"
