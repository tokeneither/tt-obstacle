"""Dense a posteriori energy estimates under exact primal/dual feasibility.

These are ordinary floating-point diagnostics, not interval-certified bounds.
They concern the fixed discrete obstacle problem, not PDE discretization error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .operators import _spd_matrix
from .reference import _vector


@dataclass(frozen=True)
class ErrorBound:
    residual_norm: float
    complementarity: float
    sharp_bound: float
    gap_bound: float


def energy_error_bound(A, f, psi, candidate, multiplier, tol=1e-12):
    """Estimate ||candidate-u_*||_A from a feasible primal/dual pair.

    With r=||A w-f-lambda||_{A^{-1}} and c=lambda^T(w-psi), the
    mathematical bounds are (r+sqrt(r*r+4*c))/2 and sqrt(r*r+2*c).

    Actual componentwise feasibility is required: even a tiny negative slack
    or multiplier raises ValueError. The tol argument controls only matrix
    symmetry validation; it does not relax feasibility or repair inputs.
    Uses a dense Cholesky solve and is intended for small diagnostic problems.
    """
    A, chol = _spd_matrix(A, "A", tol=tol)
    n = A.shape[0]
    f = _vector(f, n, "f")
    psi = _vector(psi, n, "psi")
    candidate = _vector(candidate, n, "candidate")
    multiplier = _vector(multiplier, n, "multiplier")
    slack = candidate - psi
    if np.any(slack < 0):
        raise ValueError("candidate must satisfy candidate >= psi exactly")
    if np.any(multiplier < 0):
        raise ValueError("multiplier must be componentwise nonnegative")
    residual = A @ candidate - f - multiplier
    whitened = np.linalg.solve(chol, residual)
    residual_norm = float(np.hypot.reduce(whitened))
    complementarity = float(np.dot(multiplier, slack))
    if not math.isfinite(residual_norm) or not math.isfinite(complementarity):
        raise ValueError("residual or complementarity overflowed")
    # Compute sqrt(c) without first forming products that can underflow.
    sqrt_c = float(np.hypot.reduce(np.sqrt(multiplier) * np.sqrt(slack)))
    sharp_bound = residual_norm / 2 + math.hypot(residual_norm / 2, sqrt_c)
    gap_bound = math.hypot(residual_norm, math.sqrt(2.0) * sqrt_c)
    return ErrorBound(residual_norm, complementarity, sharp_bound, gap_bound)
