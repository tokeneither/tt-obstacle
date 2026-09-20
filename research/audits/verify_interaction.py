"""Bounded numerical audits of interaction-degree compression.

Small dense arrays cross-check projections and matrices. These are floating-point
formula checks, not interval certificates and not an obstacle-PDE solver.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from verify_multidim import assemble, eigenbasis, full_operator, require
from verify_theory import tt_dense, tt_from_dense


def kron_all(factors):
    value = np.ones((1, 1)) if factors[0].ndim == 2 else np.ones(1)
    for factor in factors:
        value = np.kron(value, factor)
    return value


def apply_mode(tensor, matrix, axis):
    return np.moveaxis(np.tensordot(matrix, tensor, axes=(1, axis)), 0, axis)


def project_degree(tensor, projectors, degree):
    result = np.zeros_like(tensor)
    for mask in itertools.product([0, 1], repeat=tensor.ndim):
        if sum(mask) > degree:
            continue
        term = tensor.copy()
        for axis, active in enumerate(mask):
            p = projectors[axis]
            term = apply_mode(term, np.eye(len(p))-p if active else p, axis)
        result += term
    return result


def energy(tensor, matrix):
    flat = tensor.ravel()
    return float(flat @ matrix @ flat)


def relative_error(x, y):
    return float(np.linalg.norm(x-y)/max(np.linalg.norm(x), np.linalg.norm(y), 1e-30))


def ranks(tensor):
    return [int(np.linalg.matrix_rank(tensor.reshape(int(np.prod(tensor.shape[:k])), -1),
                                      tol=2e-9*max(np.linalg.norm(tensor), 1e-15)))
            for k in range(1, tensor.ndim)]


def local_data(n, feature_count):
    k, m = assemble(np.ones(n+1)/(n+1))
    values, vectors = eigenbasis(k, m)
    b = vectors[:, 0]
    diagonal = (m @ b)/b
    dhat = np.diag(diagonal)
    shifted = k-values[0]*m
    projection = np.outer(b, b @ m)
    require(relative_error(dhat @ b, m @ b) < 1e-12, "Shared ground-state functional")
    require(relative_error(projection, np.outer(b, b @ dhat)) < 1e-12,
            "M and Dhat ground-state projectors differ")
    scaled_m = m/np.sqrt(diagonal[:, None]*diagonal[None, :])
    c = float(np.linalg.eigvalsh(scaled_m)[0])
    require(c >= 1/3-1e-12, "Uniform-mesh comparison constant")
    raw = vectors[:, 1:feature_count+1]/b[:, None]
    probability = diagonal*b*b
    gram = raw.T @ (probability[:, None]*raw)
    gv, gu = np.linalg.eigh(gram)
    feature = raw @ ((gu/np.sqrt(gv)) @ gu.T)
    require(np.max(np.abs(probability @ feature)) < 1e-12, "Feature centering")
    require(np.linalg.norm(feature.T @ (probability[:, None]*feature)-np.eye(feature_count))
            < 1e-11, "Feature normalization")
    kappa = max(1., float(np.sqrt(np.max(np.sum(feature*feature, axis=1)))))
    feature_basis = np.column_stack([b, b[:, None]*feature])
    feature_projection = feature_basis @ np.linalg.solve(feature_basis.T @ m @ feature_basis,
                                                        feature_basis.T @ m)
    return k, m, b, dhat, shifted, values[0], projection, probability, feature, kappa, c, feature_projection


def polynomial(shape, features, degree, rng):
    tensor = np.zeros(shape)
    d = len(shape)
    for order in range(1, degree+1):
        for subset in itertools.combinations(range(d), order):
            for choices in itertools.product(range(features[0].shape[1]), repeat=order):
                factors = [np.ones(n) for n in shape]
                for axis, choice in zip(subset, choices):
                    factors[axis] = features[axis][:, choice]
                tensor += rng.normal()/np.sqrt(d**order)*kron_all(factors).reshape(shape)
    return tensor


def run_case(d, n, degree, feature_count, rng):
    shape = (n,)*d
    local = [local_data(n, feature_count) for _ in range(d)]
    ks, ms, bs, ds, ls, lambdas, ps, probabilities, features, kappas, cs, feature_ps = map(list, zip(*local))
    a = full_operator(ks, ms)
    ahat = full_operator([l+lam*dd for l, lam, dd in zip(ls, lambdas, ds)], ds)
    difference_spectrum = np.linalg.eigvalsh(ahat-a)
    scale = max(float(np.linalg.norm(ahat, 2)), 1.)
    require(difference_spectrum[0] >= -2e-10*scale, "Global A <= Ahat")
    b = kron_all(bs).reshape(shape)
    pi = kron_all(probabilities).reshape(shape)
    t = polynomial(shape, features, degree, rng)
    u = b*(1+.35*t/max(np.max(np.abs(t)), 1e-30))
    require(np.min(u/b) >= .65-1e-12, "Positive input")
    require(relative_error(u, project_degree(u, ps, degree)) < 1e-11, "Input interaction degree")

    # Actual Euclidean TT-SVD, followed by the degree projection in the theorem.
    chosen_rank = 2 if degree == 2 and feature_count == 2 else 1
    v0 = tt_dense(tt_from_dense(u, max_rank=chosen_rank))
    feature_v0 = v0.copy()
    for axis, projector in enumerate(feature_ps):
        feature_v0 = apply_mode(feature_v0, projector, axis)
    require(relative_error(v0, feature_v0) < 2e-10, "TT-SVD left the local feature spaces")
    v = project_degree(v0, ps, degree)
    before = energy(u-v0, a)
    after = energy(u-v, a)
    require(after <= before*(1+2e-8)+1e-24, "Interaction projection increased A-error")
    require(relative_error(v, project_degree(v, ps, degree)) < 2e-10, "Projection idempotence")
    require(all(rv <= (degree+1)*rv0 for rv, rv0 in zip(ranks(v), ranks(v0))),
            "Degree-projection TT rank bound")

    residual = u-v
    require(after > 1e-20, "Degenerate compression example")
    require(relative_error(residual, project_degree(residual, ps, degree)) < 2e-9,
            "Residual interaction degree")
    c, kappa = min(cs), max(kappas)
    graph_residual_energy = energy(residual, ahat)
    require(graph_residual_energy <= c**(-degree)*after*(1+2e-8),
            "Restricted mass comparison")
    transformed = residual/b
    sigma2 = float(np.sum(pi*transformed*transformed))
    envelope = b*(transformed*transformed+sigma2)/(2*np.sqrt(sigma2))
    require(np.min(envelope-np.abs(residual)) >= -1e-11, "Envelope domination")
    graph_constant = np.sqrt(1+kappa*kappa)*(kappa+3)**(degree-.5)
    full_constant = graph_constant*c**(-degree/2)
    require(energy(envelope, ahat) <= graph_constant**2*graph_residual_energy*(1+2e-8),
            "Product-graph energy theorem")
    envelope_energy = energy(envelope, a)
    require(envelope_energy <= full_constant**2*after*(1+2e-8), "Full Q1 interaction theorem")
    rr, re = ranks(residual), ranks(envelope)
    require(all(er <= 1+r*(r+1)//2 for r, er in zip(rr, re)), "Symmetric-square envelope rank")
    w = v+envelope
    # The fully contacting obstacle psi=U makes feasibility an actual constraint.
    require(np.min(w-u) >= -2e-11, "Feasible compressed output")
    require(energy(w-u, a) <= (1+full_constant)**2*after*(1+2e-8), "Final repair accuracy")
    return {"d": d, "n": n, "degree": degree, "local_feature_dimension": feature_count,
            "kappa": kappa, "c": c, "Ahat_minus_A_min_eigenvalue": float(difference_spectrum[0]),
            "actual_TT_SVD_rank_cap": chosen_rank, "projected_error_squared_over_original": after/before,
            "Ahat_residual_energy_over_A": graph_residual_energy/after,
            "restricted_comparison_bound": c**(-degree),
            "envelope_energy_amplification": float(np.sqrt(envelope_energy/after)),
            "proven_amplification": float(full_constant), "residual_ranks": rr,
            "envelope_ranks": re, "output_ranks": ranks(w),
            "minimum_feasibility_slack": float(np.min(w-u))}


def actual_rounding_counterexample(d, epsilon=.25):
    b = np.ones((2,)*d)
    h = kron_all([np.array([1., -1.])]*d).reshape((2,)*d)
    u = b+epsilon*h
    v = tt_dense(tt_from_dense(u, max_rank=1))
    require(relative_error(v, b) < 1e-12, "Rank-one TT-SVD did not return B")
    k = 3*np.array([[2., -1.], [-1., 2.]])
    m = np.array([[4., 1.], [1., 4.]])/18
    a = full_operator([k]*d, [m]*d)
    correction = epsilon*np.maximum(h, 0.)
    w = b+correction
    error = w-u
    require(np.min(w-u) >= -1e-13, "Counterexample repair feasibility for psi=U")
    require(np.min(a@error.ravel()) >= -2e-11, "Positive-correction optimality multiplier")
    beta = 6*d*(5/9)**(d-1)
    eta = 18*d*(1/3)**(d-1)
    initial_error = energy(u-v, a)
    repaired_error = energy(error, a)
    require(abs(initial_error-epsilon**2*eta) <= 1e-9*epsilon**2*eta, "Raw TT error formula")
    require(abs(repaired_error-epsilon**2*(beta+eta)/4) <= 1e-9*repaired_error,
            "Minimal positive-repair error formula")
    return {"d": d, "epsilon": epsilon, "obstacle": "psi=U=B+epsilon*checkerboard",
            "TT_SVD_result": "B (rank one)", "raw_relative_A_error": float(np.sqrt(initial_error/energy(u,a))),
            "minimal_nonnegative_repair_relative_A_error": float(np.sqrt(repaired_error/energy(u,a))),
            "repair_error_over_raw_error": float(np.sqrt(repaired_error/initial_error)),
            "signed_correction_alternative": "epsilon*checkerboard returns U exactly with TT rank <= 2"}


def main():
    rng = np.random.default_rng(20260922)
    shapes = [(2,3),(2,4),(2,5),(3,3),(3,4),(3,5),(4,3),(4,4),(5,3)]
    cases = [run_case(d, n, p, 1+(case_index % 2), rng)
             for case_index, (d,n) in enumerate(shapes) for p in [1,2]]
    counterexamples = [actual_rounding_counterexample(d) for d in [4,6,8]]
    results = {"status": "passed", "scope": "Floating-point theorem and actual TT-SVD audits; no PDE solver or interval certification",
               "interaction_cases": cases, "actual_rounding_counterexamples": counterexamples}
    Path(__file__).with_name("interaction_verification_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({"status": "passed", "interaction_cases": len(cases), "max_dense_entries": max(n**d for d,n in shapes),
                      "actual_rounding_counterexamples": len(counterexamples)}, indent=2))


if __name__ == "__main__":
    main()
