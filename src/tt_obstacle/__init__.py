"""Feasibility-preserving TT approximation for discrete obstacle problems.

Exact-arithmetic constructions are accompanied by floating-point diagnostics.
This package is research software, not an interval-certified PDE solver.
"""
from .tensor import TensorTrain
from .operators import Q1Operator, assemble_p1, generalized_eigh, uniform_p1
from .interaction import (project_degree, project_coordinates, project_spectral,
                          spectral_feature_bound, interaction_amplification_bound)
from .repair import (RepairResult, quadratic_envelope, core_envelope,
                     controlled_rank_one_envelope, repair_feasible)
from .compression import feasible_compress
from .reference import ReferenceSolution, solve_obstacle
from .certificates import ErrorBound, energy_error_bound

__version__ = "0.1.0"

__all__ = [
    "TensorTrain", "Q1Operator", "assemble_p1", "generalized_eigh", "uniform_p1",
    "project_degree", "project_coordinates", "project_spectral",
    "spectral_feature_bound", "interaction_amplification_bound",
    "RepairResult", "quadratic_envelope", "core_envelope",
    "controlled_rank_one_envelope", "repair_feasible", "feasible_compress",
    "ReferenceSolution", "solve_obstacle", "ErrorBound", "energy_error_bound",
]
