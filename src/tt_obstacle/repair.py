"""Feasibility repairs performed entirely in tensor-train form.

The inequalities in this module are exact-arithmetic statements.  Floating-point
evaluations are not interval certificates.  In particular, the optional
``structure_certified`` flag records a caller assertion; it does not establish
the local function-space assumption of the controlled rank-one theorem.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, sqrt
import operator as _index_operator

import numpy as np

from .tensor import TensorTrain


@dataclass(frozen=True)
class RepairResult:
    """A repaired tensor and directly computed energy diagnostics.

    ``error_bound`` is ``raw_error + correction_energy``.  It is a triangle
    inequality bound, not an invocation of a structural a priori constant.
    Rounding diagnostics, when present, refer to the unprojected candidate;
    an intermediate rank cap need not be respected by the repaired tensor.
    """

    tensor: TensorTrain
    candidate: TensorTrain
    correction: TensorTrain
    residual: TensorTrain
    raw_error: float
    correction_energy: float
    error_bound: float
    original_storage: int
    candidate_storage: int
    result_storage: int
    method: str
    assumptions: tuple[str, ...] = ()
    rounding_mass_error: float | None = None
    rounding_tolerance: float | None = None
    rounding_tolerance_met: bool | None = None
    rounding_ranks: tuple[int, ...] | None = None


def _degree_value(degree: int, dimension: int) -> int:
    if isinstance(degree, (bool, np.bool_)):
        raise ValueError("degree must be a nonnegative integer")
    try:
        degree = _index_operator.index(degree)
    except TypeError as exc:
        raise ValueError("degree must be a nonnegative integer") from exc
    if degree < 0:
        raise ValueError("degree must be a nonnegative integer")
    return min(degree, dimension)


def _check_shape(tensor: TensorTrain, operator) -> None:
    if tuple(tensor.shape) != tuple(operator.shape):
        raise ValueError("tensor and operator shapes must agree")


def _check_degree(tensor: TensorTrain, operator, degree: int) -> None:
    """Numerical consistency check; this is not an exact certificate."""
    from .interaction import project_degree

    degree = _degree_value(degree, len(tensor.shape))
    if degree == len(tensor.shape):
        return
    projected = project_degree(tensor, operator.projectors, degree)
    error = operator.mass_norm(tensor.subtract(projected))
    scale = operator.mass_norm(tensor)
    if error > 1.0e-10 * scale:
        raise ValueError(
            "tensor fails the numerical interaction-degree check; "
            "projecting it would change the assumed original"
        )


def quadratic_envelope(residual: TensorTrain, operator) -> TensorTrain:
    """Return ``E >= abs(residual)`` using the quadratic ground-state envelope.

    No interaction or local-space assumption is needed for domination.  Such
    assumptions are needed for the dimension-independent *energy* estimate.
    The symmetric square uses bond ranks at most ``r * (r + 1) // 2``.
    """
    _check_shape(residual, operator)
    sigma = float(operator.diagonal_mass_norm(residual))
    if sigma == 0.0:
        return TensorTrain.zeros(residual.shape)
    if not np.isfinite(sigma):
        raise ValueError("residual diagonal-mass norm must be finite")
    normalized = residual.scaled(1.0 / sigma)
    squared_over_ground = normalized.square().scale_modes(
        tuple(1.0 / b for b in operator.ground_factors)
    )
    return squared_over_ground.add(operator.ground_tensor).scaled(0.5 * sigma)


def core_envelope(residual: TensorTrain) -> TensorTrain:
    """Return an unconditional rank-one envelope from TT core slice norms.

    First left-orthogonalize the cores by QR.  Each physical slice is a matrix;
    the product of its spectral norms bounds the absolute scalar TT entry by
    submultiplicativity.  No dense tensor or structural assertion is used.
    """
    cores = [np.array(core, dtype=float, copy=True) for core in residual.cores]
    for k in range(len(cores) - 1):
        left, size, right = cores[k].shape
        q, r = np.linalg.qr(cores[k].reshape(left * size, right), mode="reduced")
        cores[k] = q.reshape(left, size, q.shape[1])
        cores[k + 1] = np.tensordot(r, cores[k + 1], axes=(1, 0))
    factors = []
    for core in cores:
        factors.append(
            np.array(
                [np.linalg.norm(core[:, i, :], ord=2) for i in range(core.shape[1])]
            )
        )
    return TensorTrain.rank_one(factors)


def rank_one_envelope(residual: TensorTrain) -> TensorTrain:
    """Alias for the unconditional :func:`core_envelope`."""
    return core_envelope(residual)


def controlled_rank_one_envelope(
    residual: TensorTrain,
    operator,
    degree: int,
    kappa: float,
    *,
    structure_certified: bool = False,
) -> TensorTrain:
    """Return the structural rank-one envelope ``sqrt(K_p) * sigma * B``.

    The caller must assert that ``residual / B`` belongs to a tensor product
    of local spaces ``span{1} + V_k``, with centered ``V_k`` satisfying the
    *whole-space* bound ``||z||_inf <= kappa ||z||_{L2(mu_k)}``, and has
    interaction degree at most ``degree``.  Here
    ``mu_k[i] = b_k[i] * (M_k @ b_k)[i]``.  A numerical degree check is made;
    the local function spaces and their bound cannot be inferred from a TT.
    ``structure_certified=True`` is the caller's assertion, not a certificate
    produced by this library.  Without these assumptions domination may fail.
    """
    _check_shape(residual, operator)
    degree = _degree_value(degree, len(residual.shape))
    if kappa is None or not np.isscalar(kappa):
        raise ValueError("kappa must be a finite real number at least one")
    try:
        kappa = float(kappa)
    except (TypeError, ValueError) as exc:
        raise ValueError("kappa must be a finite real number at least one") from exc
    if not np.isfinite(kappa) or kappa < 1.0:
        raise ValueError("kappa must be a finite real number at least one")
    if structure_certified is not True:
        raise ValueError(
            "controlled_rank_one requires structure_certified=True: "
            "the caller must assert the local-space kappa bound"
        )
    _check_degree(residual, operator, degree)
    sigma = float(operator.diagonal_mass_norm(residual))
    if sigma == 0.0:
        return TensorTrain.zeros(residual.shape)
    try:
        kernel = sum(
            comb(len(residual.shape), j) * kappa ** (2 * j)
            for j in range(degree + 1)
        )
        scale = sqrt(kernel) * sigma
    except OverflowError as exc:
        raise ValueError("controlled envelope scaling overflowed") from exc
    if not np.isfinite(scale):
        raise ValueError("controlled envelope scaling must be finite")
    return operator.ground_tensor.scaled(scale)


def repair_feasible(
    original: TensorTrain,
    candidate: TensorTrain,
    operator,
    method: str = "quadratic",
    degree: int | None = None,
    kappa: float | None = None,
    *,
    structure_certified: bool = False,
) -> RepairResult:
    """Repair a candidate so it dominates ``original`` in exact arithmetic.

    If the caller knows ``original >= obstacle``, the result is feasible for
    that obstacle.  Global feasibility of ``original`` is not checked here.
    Methods ``quadratic`` and ``core`` are unconditional.  Method
    ``controlled_rank_one`` additionally needs the explicitly asserted
    assumptions documented in :func:`controlled_rank_one_envelope`.
    """
    _check_shape(original, operator)
    _check_shape(candidate, operator)
    residual = original.subtract(candidate)
    assumptions = ()
    if method == "quadratic":
        correction = quadratic_envelope(residual, operator)
    elif method in ("core", "rank_one"):
        correction = core_envelope(residual)
    elif method == "controlled_rank_one":
        if degree is None or kappa is None:
            raise ValueError("controlled_rank_one requires degree and kappa")
        correction = controlled_rank_one_envelope(
            residual, operator, degree, kappa,
            structure_certified=structure_certified,
        )
        assumptions = (
            "Caller asserts the centered local-space kappa bound.",
            "Interaction degree was checked numerically, not certified exactly.",
        )
    else:
        raise ValueError("method must be 'quadratic', 'core', or 'controlled_rank_one'")
    tensor = candidate.add(correction)
    raw_error = float(operator.energy_norm(residual))
    correction_energy = float(operator.energy_norm(correction))
    return RepairResult(
        tensor=tensor,
        candidate=candidate,
        correction=correction,
        residual=residual,
        raw_error=raw_error,
        correction_energy=correction_energy,
        error_bound=raw_error + correction_energy,
        original_storage=original.storage_size,
        candidate_storage=candidate.storage_size,
        result_storage=tensor.storage_size,
        method=method,
        assumptions=assumptions,
    )
