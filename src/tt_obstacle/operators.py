"""Tensor-product P1/Q1 operators with consistent mass matrices.

Dense conversion is explicit and guarded. Ground-state properties additionally
require a strictly positive first eigenvector and an entrywise nonnegative mass.
Floating-point checks and norms are not interval certificates.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence

import numpy as np

from .tensor import TensorTrain


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _finite_real(value, name):
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real")
    try:
        result = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain real numbers") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _spd_matrix(matrix, name="matrix", tol=1e-12):
    """Validate, copy, and symmetrize within the stated relative tolerance."""
    array = _finite_real(matrix, name)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative")
    scale = float(np.max(np.abs(array)))
    if np.max(np.abs(array - array.T)) > tol * scale:
        raise ValueError(f"{name} must be symmetric")
    array = array / 2 + array.T / 2
    try:
        chol = np.linalg.cholesky(array)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite") from exc
    return array, chol


def assemble_p1(nodes):
    """Return interior Dirichlet stiffness and consistent mass on a 1D mesh.

    Nodes includes both boundary points and at least one interior point.
    The resulting matrices use the usual piecewise linear nodal basis.
    """
    nodes = _finite_real(nodes, "nodes")
    if nodes.ndim != 1 or nodes.size < 3:
        raise ValueError("nodes must include two boundaries and an interior point")
    intervals = np.diff(nodes)
    if not np.all(np.isfinite(intervals)) or np.any(intervals <= 0):
        raise ValueError("nodes must be strictly increasing with finite spacings")
    n = nodes.size - 2
    inverse = 1.0 / intervals
    stiffness = np.diag(inverse[:-1] + inverse[1:])
    mass = np.diag((intervals[:-1] + intervals[1:]) / 3.0)
    if n > 1:
        stiffness += np.diag(-inverse[1:-1], 1) + np.diag(-inverse[1:-1], -1)
        mass += np.diag(intervals[1:-1] / 6.0, 1)
        mass += np.diag(intervals[1:-1] / 6.0, -1)
    if not np.all(np.isfinite(stiffness)):
        raise ValueError("mesh spacing is too small for finite stiffness entries")
    return stiffness, mass


def uniform_p1(n, length=1.0):
    """Return P1 matrices on n interior nodes of an interval of given length."""
    n = _positive_integer(n, "n")
    length_array = _finite_real(length, "length")
    if length_array.ndim != 0 or length_array <= 0:
        raise ValueError("length must be a positive finite scalar")
    return assemble_p1(np.linspace(0.0, float(length_array), n + 2))


def generalized_eigh(stiffness, mass):
    """Solve K v = lambda M v with M-orthonormal eigenvector columns."""
    stiffness, _ = _spd_matrix(stiffness, "stiffness")
    mass, chol = _spd_matrix(mass, "mass")
    if stiffness.shape != mass.shape:
        raise ValueError("stiffness and mass must have equal shapes")
    transformed = np.linalg.solve(chol, stiffness)
    transformed = np.linalg.solve(chol, transformed.T).T
    transformed = transformed / 2 + transformed.T / 2
    eigenvalues, orthogonal = np.linalg.eigh(transformed)
    eigenvectors = np.linalg.solve(chol.T, orthogonal)
    for k in range(eigenvectors.shape[1]):
        column = eigenvectors[:, k]
        if column[np.argmax(np.abs(column))] < 0:
            eigenvectors[:, k] *= -1
    return eigenvalues, eigenvectors


def _advance(environment, left_core, right_core, factor):
    transformed = np.einsum("ij,bjd->bid", factor, right_core, optimize=True)
    return np.einsum(
        "ab,aic,bid->cd", environment, left_core, transformed, optimize=True
    )


def _readonly(array):
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


class Q1Operator:
    """Separable Dirichlet operator sum_k K_k tensor M_others.

    Any symmetric positive definite local factors are accepted for energy/mass
    operations. Ground-state APIs reject factors outside their additional
    positivity assumptions instead of silently assuming a Q1 comparison theorem.
    """

    def __init__(self, stiffness: Sequence, mass: Sequence):
        try:
            stiffness = tuple(stiffness)
            mass = tuple(mass)
        except TypeError as exc:
            raise ValueError("stiffness and mass must be nonempty sequences") from exc
        if not stiffness or len(stiffness) != len(mass):
            raise ValueError("stiffness and mass need the same positive length")
        ks, ms, k_chols, m_chols, values, vectors = [], [], [], [], [], []
        for k, (local_k, local_m) in enumerate(zip(stiffness, mass)):
            local_k, chol_k = _spd_matrix(local_k, f"stiffness[{k}]")
            local_m, chol_m = _spd_matrix(local_m, f"mass[{k}]")
            if local_k.shape != local_m.shape:
                raise ValueError(f"local factor shapes disagree in mode {k}")
            eigvals, eigvecs = generalized_eigh(local_k, local_m)
            ks.append(_readonly(local_k))
            ms.append(_readonly(local_m))
            k_chols.append(_readonly(chol_k))
            m_chols.append(_readonly(chol_m))
            values.append(_readonly(eigvals))
            vectors.append(_readonly(eigvecs))
        self.stiffness = tuple(ks)
        self.mass = tuple(ms)
        self._k_chols = tuple(k_chols)
        self._m_chols = tuple(m_chols)
        self.eigenvalues = tuple(values)
        self.eigenvectors = tuple(vectors)
        self.shape = tuple(matrix.shape[0] for matrix in self.mass)
        self.dimension = len(self.shape)

    @classmethod
    def uniform(cls, shape, lengths=None):
        """Build a tensor-product Q1 operator on uniform 1D meshes."""
        try:
            shape = tuple(_positive_integer(n, "shape entry") for n in shape)
        except TypeError as exc:
            raise ValueError("shape must be a nonempty sequence") from exc
        if not shape:
            raise ValueError("shape must be nonempty")
        if lengths is None:
            lengths = (1.0,) * len(shape)
        try:
            lengths = tuple(lengths)
        except TypeError as exc:
            raise ValueError("lengths must have one entry per mode") from exc
        if len(lengths) != len(shape):
            raise ValueError("lengths must have one entry per mode")
        factors = [uniform_p1(n, length) for n, length in zip(shape, lengths)]
        return cls([pair[0] for pair in factors], [pair[1] for pair in factors])

    @property
    def ground_factors(self):
        factors = tuple(vectors[:, 0] for vectors in self.eigenvectors)
        if any(np.any(factor <= 0) for factor in factors):
            raise ValueError("ground-state APIs require a strictly positive first eigenvector")
        if any(np.any(matrix < 0) for matrix in self.mass):
            raise ValueError("ground-state APIs require entrywise nonnegative mass matrices")
        return factors

    @property
    def ground_eigenvalues(self):
        return _readonly([values[0] for values in self.eigenvalues])

    @property
    def ground_tensor(self):
        return TensorTrain.rank_one(self.ground_factors)

    @property
    def projectors(self):
        return tuple(
            _readonly(np.outer(b, matrix @ b))
            for b, matrix in zip(self.ground_factors, self.mass)
        )

    @property
    def diagonal_mass_factors(self):
        return tuple(
            _readonly((matrix @ b) / b)
            for b, matrix in zip(self.ground_factors, self.mass)
        )

    @property
    def comparison_constant(self):
        """Largest common c with c Dhat_k <= M_k, computed locally."""
        constants = []
        for mass, diagonal in zip(self.mass, self.diagonal_mass_factors):
            inverse_sqrt = 1.0 / np.sqrt(diagonal)
            transformed = inverse_sqrt[:, None] * mass * inverse_sqrt[None, :]
            constants.append(float(np.linalg.eigvalsh(transformed)[0]))
        return min(constants)

    def _validate_tensor(self, tensor):
        if not isinstance(tensor, TensorTrain):
            raise TypeError("expected a TensorTrain")
        if tuple(tensor.shape) != self.shape:
            raise ValueError("tensor shape does not match the operator")

    def mass_inner(self, left, right):
        """Compute left^T M right by TT contractions, without dense tensors."""
        self._validate_tensor(left)
        self._validate_tensor(right)
        environment = np.ones((1, 1))
        for a, b, mass in zip(left.cores, right.cores, self.mass):
            environment = _advance(environment, a, b, mass)
        return float(environment[0, 0])

    def energy_inner(self, left, right):
        """Compute left^T A right in one sweep over TT cores.

        Two environments carry the product mass and the accumulated sum of
        stiffness contributions. Floating-point cancellation is possible for
        signed bilinear forms; use energy_norm for stable self norms.
        """
        self._validate_tensor(left)
        self._validate_tensor(right)
        mass_environment = np.ones((1, 1))
        energy_environment = np.zeros((1, 1))
        for a, b, stiffness, mass in zip(
            left.cores, right.cores, self.stiffness, self.mass
        ):
            energy_environment = (
                _advance(energy_environment, a, b, mass)
                + _advance(mass_environment, a, b, stiffness)
            )
            mass_environment = _advance(mass_environment, a, b, mass)
        return float(energy_environment[0, 0])

    def mass_norm(self, tensor):
        """Stable norm via local Cholesky whitening and the TT QR norm."""
        self._validate_tensor(tensor)
        return float(tensor.mode_product([chol.T for chol in self._m_chols]).norm())

    def diagonal_mass_norm(self, tensor):
        """Stable product-Dhat norm via local diagonal whitening and TT QR."""
        self._validate_tensor(tensor)
        factors = [np.sqrt(diagonal) for diagonal in self.diagonal_mass_factors]
        return float(tensor.scale_modes(factors).norm())

    def energy_norm(self, tensor):
        """Stable energy norm from whitened local terms and hypot accumulation.

        This uses d stable TT norms rather than taking a square root of a
        potentially cancelling self-contraction. No global matrix is formed.
        """
        self._validate_tensor(tensor)
        result = 0.0
        for k in range(self.dimension):
            whiteners = [chol.T for chol in self._m_chols]
            whiteners[k] = self._k_chols[k].T
            result = math.hypot(result, float(tensor.mode_product(whiteners).norm()))
        return result

    def _check_dense_limit(self, max_entries):
        max_entries = _positive_integer(max_entries, "max_entries")
        unknowns = math.prod(self.shape)
        if unknowns * unknowns > max_entries:
            raise ValueError(
                f"dense matrix would have {unknowns * unknowns} entries, "
                f"exceeding max_entries={max_entries}"
            )
        return unknowns

    def mass_dense(self, max_entries=1_000_000):
        """Form the global mass matrix only after checking its entry count."""
        self._check_dense_limit(max_entries)
        result = np.ones((1, 1))
        for factor in self.mass:
            result = np.kron(result, factor)
        return result

    def to_dense(self, max_entries=1_000_000):
        """Form the global stiffness matrix with a matrix-entry-count guard."""
        unknowns = self._check_dense_limit(max_entries)
        result = np.zeros((unknowns, unknowns))
        for k in range(self.dimension):
            term = np.ones((1, 1))
            for j in range(self.dimension):
                factor = self.stiffness[j] if j == k else self.mass[j]
                term = np.kron(term, factor)
            result += term
        return result

