"""Small independent dense obstacle reference by exhaustive active sets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .operators import _finite_real, _positive_integer, _spd_matrix


@dataclass(frozen=True)
class ReferenceSolution:
    """Raw primal solution and multiplier A u - f; neither is clipped."""

    solution: np.ndarray
    multiplier: np.ndarray


def _vector(vector, n, name):
    result = _finite_real(vector, name)
    if result.ndim != 1 or result.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},)")
    return result


def solve_obstacle(A, f, psi, max_unknowns=16, tol=1e-10):
    """Enumerate active sets to solve min_{u>=psi} .5 u^T A u - f^T u.

    This is an exponential-cost validation reference, not a scalable solver.
    The guard is checked before factorization or enumeration. KKT checks allow
    the supplied absolute tolerance, but returned arrays are never clipped;
    users must check actual feasibility before requesting a strict error bound.
    """
    max_unknowns = _positive_integer(max_unknowns, "max_unknowns")
    raw = _finite_real(A, "A")
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1] or raw.shape[0] == 0:
        raise ValueError("A must be a nonempty square matrix")
    n = raw.shape[0]
    if n > max_unknowns:
        raise ValueError(f"{n} unknowns exceed max_unknowns={max_unknowns}")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative")
    A, _ = _spd_matrix(raw, "A", tol=tol)
    f = _vector(f, n, "f")
    psi = _vector(psi, n, "psi")
    indices = np.arange(n)
    for mask in range(1 << n):
        active = ((mask >> indices) & 1).astype(bool)
        free = ~active
        solution = np.array(psi, copy=True)
        if np.any(free):
            rhs = f[free] - A[np.ix_(free, active)] @ psi[active]
            solution[free] = np.linalg.solve(A[np.ix_(free, free)], rhs)
        multiplier = A @ solution - f
        slack = solution - psi
        if (
            np.all(slack >= -tol)
            and np.all(multiplier >= -tol)
            and np.all(np.abs(multiplier[free]) <= tol)
            and np.all(np.abs(slack * multiplier) <= tol)
        ):
            solution.setflags(write=False)
            multiplier.setflags(write=False)
            return ReferenceSolution(solution, multiplier)
    raise RuntimeError("no active set satisfies KKT within tol; adjust scaling/tolerance")

