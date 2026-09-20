"""Small independent convex-QP audits of solution approximation theorems.

Run python verify_solution_structure.py. These checks use
floating-point arithmetic and are not interval certificates. Large dimensions
occur only in exact reduced counterexample formulas, never in dense full cubes.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np

from verify_theory import solve_obstacle
from verify_multidim import assemble, eigenbasis, full_operator, require
from verify_interaction import kron_all, project_degree, energy, relative_error


def assert_leq(left, right, label, scale=1.):
    require(left <= right+2e-8*max(abs(left), abs(right), scale), label)


def kron_coordinate(op, axis, shape):
    return kron_all([op if j == axis else np.eye(n) for j, n in enumerate(shape)])


def check_qp(shape, degree, rng):
    d = len(shape)
    ks, ms, vs, vals, ps, gs = [], [], [], [], [], []
    for n in shape:
        k, mass = assemble(np.ones(n+1)/(n+1))
        values, vectors = eigenbasis(k, mass)
        b = vectors[:, 0]
        projection = np.outer(b, b @ mass)
        m1 = float(b @ mass @ b / (b @ b))
        graph = (k-values[0]*mass)/m1
        ks.append(k)
        ms.append(mass)
        vs.append(vectors)
        vals.append(values)
        ps.append(projection)
        gs.append(graph)
    a = full_operator(ks, ms)
    transform = kron_all(vs)
    coeff = rng.normal(size=shape)
    if degree is not None:
        for index in np.ndindex(shape):
            if sum(j != 0 for j in index) > degree:
                coeff[index] = 0.
    coeff[(0,)*d] = .1*rng.normal()
    y = transform @ coeff.ravel()
    y /= max(1., np.max(np.abs(y)))
    f = a @ y
    v, lam = solve_obstacle(a, f, np.zeros(len(y)))
    kkt = a @ v-f
    require(np.min(v) >= -1e-10 and np.min(kkt) >= -1e-8, "QP primal/dual feasibility")
    require(np.max(np.abs(v*kkt)) < 2e-8, "QP complementarity")
    require(np.linalg.norm(kkt-lam) < 2e-8, "Independent multiplier residual")
    ey, ev = energy(y, a), energy(v, a)
    require(ev > 1e-12, "Trivial zero solution")
    sy = sv = 0.
    cylinder = []
    for axis in range(d):
        projection = kron_coordinate(ps[axis], axis, shape)
        dy, dv = y-projection@y, v-projection@v
        sy += energy(dy, a)
        sv += energy(dv, a)
        projected_y = projection @ y
        vi, _ = solve_obstacle(a, a @ projected_y, np.zeros(len(y)))
        require(relative_error(vi, projection@vi) < 5e-8 or np.linalg.norm(vi) < 1e-10,
                "Cylinder solution left invariant cylinder")
        err = energy(v-vi, a)
        tail = energy(dy, a)
        assert_leq(err, tail, "Cylinder error exceeds data tail")
        assert_leq(energy(dv, a), tail, "Coordinate solution tail exceeds data tail")
        cylinder.append(float(err/max(tail, 1e-30)))
    assert_leq(sv, sy, "Total influence increased")
    anova_tail = []
    for p in range(d):
        projected = project_degree(v.reshape(shape), ps, p).ravel()
        tail = energy(v-projected, a)
        assert_leq((p+1)*tail, sy, "ANOVA tail exceeds influence bound")
        if degree is not None:
            assert_leq((p+1)*tail, degree*ey, "Low-degree data tail bound")
        anova_tail.append(float((p+1)*tail/max(sy, 1e-30)))

    g = sum(kron_coordinate(graph, axis, shape) for axis, graph in enumerate(gs))
    ag = a @ g
    require(np.linalg.norm(ag-ag.T) < 5e-10*max(1., np.linalg.norm(ag)), "A and graph G do not commute")
    ry, rv = float(y @ ag @ y), float(v @ ag @ v)
    assert_leq(rv, ry, "Commuting positive-semigroup AG regularity increased")
    spectral = []
    for keep in (1, 2):
        factors, thresholds = [], []
        for n, vectors, mass, graph in zip(shape, vs, ms, gs):
            use = min(keep, n)
            basis = vectors[:, :use]
            factors.append(basis @ basis.T @ mass)
            if use < n:
                column = vectors[:, use]
                thresholds.append(float(column @ graph @ column/(column @ column)))
        if not thresholds:
            continue
        threshold = min(thresholds)
        truncated = kron_all(factors) @ v
        tail = energy(v-truncated, a)
        assert_leq(threshold*tail, ry, "Fixed-mode spectral tail estimate")
        spectral.append({"retained_modes": keep, "threshold": threshold,
                         "bound_ratio": float(threshold*tail/max(ry, 1e-30))})
    return {"shape": list(shape), "data_interaction_degree": degree,
            "active_nodes": int(np.sum(v < 1e-8)), "nodes": len(v),
            "max_complementarity_residual": float(np.max(np.abs(v*kkt))),
            "solution_energy_over_free_energy": ev/ey,
            "influence_solution_over_data": sv/max(sy, 1e-30),
            "cylinder_error_over_data_tail": cylinder,
            "degree_tail_bound_ratios": anova_tail,
            "AG_regularity_solution_over_data": rv/max(ry, 1e-30),
            "spectral_tail_checks": spectral}


def check_counterexample(m):
    d = 6*m
    k = 3*np.array([[2., -1.], [-1., 2.]])
    mass = np.array([[4., 1.], [1., 4.]])/18
    mu, rho, lam0 = 5/18, 3/5, 54/5
    a = full_operator([k]*m, [mass]*m)+(d-m)*lam0*kron_all([mass]*m)
    require(np.min(a) >= -1e-14, "Reduced matrix has negative entries")
    signs = np.array(list(itertools.product([1., -1.], repeat=m)))
    rhs = np.sum(signs, axis=1)-m+2
    u = np.zeros(2**m)
    u[0] = 2/a[0, 0]
    multiplier = a @ u-rhs
    require(np.min(multiplier) >= -1e-12, "Counterexample dual feasibility")
    require(np.max(np.abs(u*multiplier)) < 1e-10, "Counterexample complementarity")
    require(np.min(multiplier[1:]) > 0., "Counterexample strict contact multiplier")
    if m <= 3:
        solved, _ = solve_obstacle(a, rhs, np.zeros(2**m))
        require(relative_error(u, solved) < 1e-10, "Counterexample differs from independent QP solve")
    y = np.linalg.solve(a, rhs)
    ratio = energy(u, a)/energy(y, a)
    formula = 4/((8/5)**m*(1+3*m/(2*d))*((m-2)**2+5*m/(3*(1+4/d))))
    require(abs(ratio-formula) < 2e-10*max(ratio, formula), "Conditioning formula")
    hadamard = kron_all([np.array([[1., 1.], [1., -1.]])]*m)
    coeff = hadamard.T @ u/2**m
    order = np.array([sum(index) for index in itertools.product([0, 1], repeat=m)])
    a0 = mu**m*lam0*d
    expected = a0*rho**order*(1+4*order/d)
    actual = np.diag(hadamard.T @ a @ hadamard)/2**m
    require(np.max(np.abs(actual-expected)) < 2e-10*max(expected), "Reduced spectral formula")
    total = energy(u, a)
    tails = []
    for p in [0, 1, min(2, m-1)]:
        truncated = hadamard @ (coeff*(order <= p))
        tail = energy(u-truncated, a)/total
        expected_tail = sum(math.comb(m, j)*rho**j*(1+4*j/d) for j in range(p+1, m+1))
        expected_tail /= (1+rho)**m*(1+3*m/(2*d))
        require(abs(tail-expected_tail) < 2e-10, "Counterexample interaction tail formula")
        tails.append({"p": p, "relative_tail_squared": tail})
    return {"m": m, "d": d, "reduced_nodes": 2**m,
            "minimum_contact_multiplier": float(np.min(multiplier[1:])),
            "solution_energy_over_free_energy": ratio,
            "interaction_tails": tails}


def main():
    rng = np.random.default_rng(202609201)
    cases = [((2, 2), 1), ((2, 2), None), ((2, 3), 1), ((2, 3), None),
             ((3, 3), 1), ((3, 3), None), ((2, 2, 2), 1), ((2, 2, 2), 2),
             ((2, 2, 2), None), ((2, 2, 3), 1), ((2, 2, 3), 2), ((2, 2, 3), None)]
    qp = [check_qp(shape, degree, rng) for shape, degree in cases]
    counter = [check_counterexample(m) for m in (2, 3, 4, 5, 6)]
    asymptotic = []
    for m in (10, 20, 40, 80, 160):
        d, p = 6*m, 2
        captured = sum(math.comb(m, k)*(3/5)**k*(1+4*k/d) for k in range(p+1))
        captured /= (8/5)**m*(1+3*m/(2*d))
        asymptotic.append({"m": m, "d": d, "p": p, "captured_energy_fraction": captured})
    output = {"status": "passed", "floating_point_not_interval_certificate": True,
              "independent_QP_cases": len(qp), "maximum_full_QP_nodes": max(c["nodes"] for c in qp),
              "QP_cases": qp, "reduced_counterexample_cases": counter,
              "analytic_large_dimension_tail_values": asymptotic}
    path = Path(__file__).with_name("solution_structure_results.json")
    path.write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps({"status": output["status"], "QP_cases": len(qp),
                      "reduced_counterexamples": len(counter), "output": str(path)}, indent=2))


if __name__ == "__main__":
    main()
