"""Ground-state interaction projectors implemented directly on TT cores."""
from __future__ import annotations

import math
from numbers import Integral

import numpy as np

from .tensor import TensorTrain


def _projectors(tensor, projectors):
    raw = tuple(np.asarray(p) for p in projectors)
    if any(np.iscomplexobj(p) for p in raw):
        raise ValueError("Complex projectors are not supported.")
    projectors = tuple(np.asarray(p, dtype=float) for p in raw)
    if len(projectors) != len(tensor.shape):
        raise ValueError("Provide one projector per tensor coordinate.")
    for n, p in zip(tensor.shape, projectors):
        if p.shape != (n, n) or not np.all(np.isfinite(p)):
            raise ValueError("Projectors must be finite square matrices matching the modes.")
        if not np.allclose(p @ p, p, rtol=1e-10, atol=1e-12):
            raise ValueError("Each local matrix must be a projector (P @ P = P).")
    return projectors


def project_degree(tensor: TensorTrain, projectors, degree: int) -> TensorTrain:
    """Keep interactions of order at most ``degree`` relative to local P's.

    The counting-automaton MPO increases each TT bond by at most degree + 1.
    With Q1 ground-state projectors it is orthogonal in the Q1 energy norm.
    It is generally NOT positivity preserving. No dense tensor is assembled.
    """
    if isinstance(degree, bool) or not isinstance(degree, Integral) or degree < 0:
        raise ValueError("degree must be a nonnegative integer.")
    ps = _projectors(tensor, projectors)
    d = len(tensor.shape)
    if degree >= d:
        return tensor.copy()
    result = []
    for axis, (core, p) in enumerate(zip(tensor.cores, ps)):
        left_rank, n, right_rank = core.shape
        pcore = np.einsum("ij,ajb->aib", p, core, optimize=True)
        qcore = core - pcore
        left_states = min(degree, axis) + 1
        last = axis == d - 1
        right_states = 1 if last else min(degree, axis + 1) + 1
        out = np.zeros((left_states * left_rank, n, right_states * right_rank))
        for count in range(left_states):
            rows = slice(count * left_rank, (count + 1) * left_rank)
            target = 0 if last else count
            cols = slice(target * right_rank, (target + 1) * right_rank)
            out[rows, :, cols] += pcore
            if count < degree:
                target = 0 if last else count + 1
                cols = slice(target * right_rank, (target + 1) * right_rank)
                out[rows, :, cols] += qcore
        result.append(out)
    return TensorTrain(result)


def project_coordinates(tensor: TensorTrain, projectors, keep) -> TensorTrain:
    """Average out coordinates not in ``keep``; zero-based coordinate indices.

    For positive ground-state projectors this preserves nonnegativity. It does
    not preserve an arbitrary nonconstant obstacle without first shifting it.
    """
    ps = _projectors(tensor, projectors)
    keep = tuple(keep)
    if any(isinstance(k, bool) or not isinstance(k, Integral)
           or k < 0 or k >= len(tensor.shape) for k in keep):
        raise ValueError("keep must contain valid zero-based coordinate indices.")
    if len(set(keep)) != len(keep):
        raise ValueError("keep must not contain duplicate coordinates.")
    chosen = set(keep)
    return tensor.mode_product([np.eye(n) if k in chosen else ps[k]
                                for k, n in enumerate(tensor.shape)])


def project_spectral(tensor: TensorTrain, operator, num_modes: int) -> TensorTrain:
    """Keep up to ``num_modes`` generalized eigenmodes per coordinate.

    Ground state counts as one mode. This linear projector need not preserve
    an obstacle or nonnegativity.
    """
    if isinstance(num_modes, bool) or not isinstance(num_modes, Integral) or num_modes < 1:
        raise ValueError("num_modes must be a positive integer.")
    if tensor.shape != operator.shape:
        raise ValueError("Tensor and Q1 operator shapes differ.")
    matrices = []
    for vectors, mass in zip(operator.eigenvectors, operator.mass):
        basis = vectors[:, :num_modes]
        matrices.append(basis @ basis.T @ mass)
    return tensor.mode_product(matrices)


def spectral_feature_bound(operator, num_modes: int) -> float:
    """Whole-space kappa bound for ratios of the first eigenmodes to b.

    The returned value applies only if a tensor lies in the stated product
    space. It does not certify membership of an arbitrary tensor. At most
    ``num_modes`` physical generalized eigenmodes (including b) are used.
    Computed in floating point, not by interval arithmetic.
    """
    if isinstance(num_modes, bool) or not isinstance(num_modes, Integral) or num_modes < 1:
        raise ValueError("num_modes must be a positive integer.")
    bound = 1.0
    for b, mass, vectors in zip(operator.ground_factors, operator.mass,
                                operator.eigenvectors):
        raw = vectors[:, 1:num_modes] / b[:, None]
        if raw.shape[1] == 0:
            continue
        probability = b * (mass @ b)
        gram = raw.T @ (probability[:, None] * raw)
        values, rotation = np.linalg.eigh(gram)
        if np.any(values <= 0):
            raise ValueError("Local feature Gram matrix lost positive definiteness.")
        features = raw @ (rotation / np.sqrt(values))
        bound = max(bound, float(np.sqrt(np.max(np.sum(features ** 2, axis=1)))))
    return bound


def interaction_amplification_bound(degree: int, kappa: float, comparison: float) -> float:
    """Conditional Q1 bound ||E_quadratic||_A / ||R||_A from the theory.

    Requires residual interaction order <= degree and the WHOLE local-space
    L-infinity/L2 bound kappa. The graph proof also requires nonpositive
    off-diagonal entries of K_k - lambda_{k,1} M_k, as for standard P1
    stiffness/mass matrices. Arbitrary SPD factors accepted by Q1Operator
    need not satisfy this condition. This function evaluates a constant,
    not a certificate of the operator or structural assumptions.
    """
    if isinstance(degree, bool) or not isinstance(degree, Integral) or degree < 1:
        raise ValueError("The interaction energy theorem requires degree >= 1.")
    if not math.isfinite(kappa) or kappa < 1:
        raise ValueError("kappa must be finite and at least one.")
    if not math.isfinite(comparison) or not 0 < comparison <= 1:
        raise ValueError("comparison must lie in (0, 1].")
    return math.sqrt(1 + kappa * kappa) * (kappa + 3) ** (degree - .5) * comparison ** (-degree / 2)
