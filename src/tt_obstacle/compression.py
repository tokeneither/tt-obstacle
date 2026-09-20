"""Mass-weighted TT rounding, optional degree projection, and feasibility repair."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .interaction import project_degree
from .repair import RepairResult, _check_degree, _check_shape, _degree_value, repair_feasible
from .tensor import TensorTrain


def feasible_compress(
    original: TensorTrain,
    operator,
    rtol: float = 1.0e-8,
    max_rank=None,
    degree: int | None = None,
    method: str = "quadratic",
    kappa: float | None = None,
    *,
    structure_certified: bool = False,
) -> RepairResult:
    """Build a feasible approximation without materializing the full tensor.

    The input is mapped by ``chol(M_k).T`` into Euclidean coordinates,
    rounded there, and mapped back.  ``rtol`` is therefore a *mass-norm*
    tolerance.  A rank cap can prevent meeting it; the returned rounding
    diagnostics report this explicitly.  ``max_rank`` caps only this initial
    rounding, not subsequent degree projection or feasibility repair.

    When ``degree`` is supplied, the original's membership is checked to a
    relative mass-norm tolerance of ``1e-10`` before projecting the candidate.
    This floating-point check is not an exact structural certificate.  The
    controlled rank-one method additionally requires the caller assertion
    ``structure_certified=True`` and the whole-space local ``kappa`` bound.
    Input feasibility is assumed: if ``original >= obstacle``, the repaired
    result satisfies that obstacle under the repair method's assumptions.
    """
    _check_shape(original, operator)
    if isinstance(rtol, (bool, np.bool_)):
        raise ValueError("rtol must be a finite nonnegative real number")
    try:
        rtol = float(rtol)
    except (TypeError, ValueError) as exc:
        raise ValueError("rtol must be a finite nonnegative real number") from exc
    if not np.isfinite(rtol) or rtol < 0.0:
        raise ValueError("rtol must be a finite nonnegative real number")
    if degree is not None:
        degree = _degree_value(degree, len(original.shape))
        _check_degree(original, operator, degree)
    if method == "controlled_rank_one" and (degree is None or kappa is None):
        raise ValueError("controlled_rank_one requires degree and kappa")

    whiteners = tuple(np.linalg.cholesky(m).T for m in operator.mass)
    inverses = tuple(np.linalg.solve(w, np.eye(w.shape[0])) for w in whiteners)
    whitened = original.mode_product(whiteners)
    rounded = whitened.round(rtol=rtol, max_rank=max_rank)
    candidate = rounded.mode_product(inverses)
    rounding_error = float(operator.mass_norm(original.subtract(candidate)))
    mass_scale = float(operator.mass_norm(original))
    tolerance = rtol * mass_scale
    # Allow only a floating-point floor, not a hidden approximation tolerance.
    tolerance_met = rounding_error <= tolerance + 128.0 * np.finfo(float).eps * mass_scale
    if degree is not None:
        candidate = project_degree(candidate, operator.projectors, degree)
    result = repair_feasible(
        original, candidate, operator, method=method, degree=degree, kappa=kappa,
        structure_certified=structure_certified,
    )
    return replace(
        result,
        rounding_mass_error=rounding_error,
        rounding_tolerance=tolerance,
        rounding_tolerance_met=bool(tolerance_met),
        rounding_ranks=tuple(rounded.ranks),
    )
