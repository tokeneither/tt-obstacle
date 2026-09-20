"""Small numerical audits of the accompanying proofs, not a PDE benchmark.

Run with python verify_theory.py. Dense reference arrays are
used only on small problems. No floating-point computation here is an interval
certificate; the rigorous statements are the exact-arithmetic proofs.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import numpy as np


def norm_a(x, a):
    return float(np.sqrt(max(0.0, x @ a @ x)))


def phi(r, c):
    assert c > -1e-9
    return (r + np.sqrt(r*r + 4*max(c, 0.0))) / 2


def solve_obstacle(a, f, psi):
    """Enumerate all active sets: independent tiny convex-QP reference."""
    n = len(f)
    for bits in itertools.product([False, True], repeat=n):
        active = np.array(bits)
        free = ~active
        x = psi.copy()
        if np.any(free):
            x[free] = np.linalg.solve(a[np.ix_(free, free)],
                f[free] - a[np.ix_(free, active)] @ x[active])
        lam = a @ x - f
        if (np.all(x >= psi-1e-9) and np.all(lam[active] >= -1e-9)
                and np.all(np.abs(lam[free]) <= 1e-8)):
            lam[free] = 0.0
            return x, np.maximum(lam, 0)
    raise AssertionError("No KKT solution found")


def certificate(a, f, psi, w, lam):
    assert np.min(w-psi) > -1e-9 and np.min(lam) > -1e-9
    residual = a @ w - f - lam
    r = np.sqrt(max(0.0, residual @ np.linalg.solve(a, residual)))
    c = float(lam @ (w-psi))
    return float(phi(r, c)), float(np.sqrt(r*r+2*max(c, 0))), r, c


def tt_dense(cores):
    x = cores[0][0]
    for g in cores[1:]:
        x = np.tensordot(x, g, axes=(-1, 0))
    return x[..., 0]


def tt_from_dense(x, max_rank=None):
    dims = x.shape
    rprev = 1
    rem = x.copy()
    cores = []
    for k, n in enumerate(dims[:-1]):
        u, s, vt = np.linalg.svd(rem.reshape(rprev*n, -1), full_matrices=False)
        rnew = len(s) if max_rank is None else min(max_rank, len(s))
        cores.append(u[:, :rnew].reshape(rprev, n, rnew))
        rem = s[:rnew, None] * vt[:rnew]
        rprev = rnew
    cores.append(rem.reshape(rprev, dims[-1], 1))
    return cores


def tt_add(a, b, sign=1.0):
    if len(a) == 1:
        return [a[0] + sign*b[0]]
    result = [np.concatenate([a[0], sign*b[0]], axis=2)]
    for ga, gb in zip(a[1:-1], b[1:-1]):
        g = np.zeros((ga.shape[0]+gb.shape[0], ga.shape[1], ga.shape[2]+gb.shape[2]))
        g[:ga.shape[0], :, :ga.shape[2]] = ga
        g[ga.shape[0]:, :, ga.shape[2]:] = gb
        result.append(g)
    result.append(np.concatenate([a[-1], b[-1]], axis=0))
    return result


def tt_scale_separable(cores, factors):
    return [g * b[None, :, None] for g, b in zip(cores, factors)]


def tt_left_orthogonalize(cores):
    gs = [g.copy() for g in cores]
    for k in range(len(gs)-1):
        rleft, n, rright = gs[k].shape
        q, r = np.linalg.qr(gs[k].reshape(rleft*n, rright), mode="reduced")
        gs[k] = q.reshape(rleft, n, q.shape[1])
        gs[k+1] = np.einsum("ab,bic->aic", r, gs[k+1])
    return gs


def tt_product_inner(a, b, matrices=None):
    z = np.ones((1, 1))
    if matrices is None:
        matrices = [np.eye(g.shape[1]) for g in a]
    for ga, gb, m in zip(a, b, matrices):
        z = np.einsum("ab,aic,ij,bjd->cd", z, ga, m, gb, optimize=True)
    return float(z[0, 0])


def tt_norm(cores):
    # Orthogonalization avoids cancellation in the final norm contraction.
    return float(np.linalg.norm(tt_left_orthogonalize(cores)[-1]))


def rank_one_envelope(cores):
    """Product of spectral norms of slices of left-canonical TT cores."""
    gs = tt_left_orthogonalize(cores)
    return [np.array([np.linalg.norm(g[:, i, :], 2)
            for i in range(g.shape[1])]) for g in gs]


def separable_dense(factors):
    z = factors[0]
    for b in factors[1:]:
        z = np.multiply.outer(z, b)
    return z


def fem_1d(n):
    h = 1/(n+1)
    k = (2*np.eye(n)-np.eye(n, k=1)-np.eye(n, k=-1))/h
    m = h*(4*np.eye(n)+np.eye(n, k=1)+np.eye(n, k=-1))/6
    x = np.arange(1, n+1)*h
    return k, m, x*(1-x)


def tt_energy(cores, stiffness, mass):
    total = 0.0
    for j in range(len(cores)):
        mats = [stiffness[k] if k == j else mass[k] for k in range(len(cores))]
        total += tt_product_inner(cores, cores, mats)
    return float(np.sqrt(max(total, 0.0)))


def run_audits():
    rng = np.random.default_rng(20260920)
    result = {}
    ratios = []
    for _ in range(120):
        n = int(rng.integers(2, 7))
        z = rng.normal(size=(n, n))
        a = z.T@z + 0.5*np.eye(n)
        psi, f = rng.normal(size=n), rng.normal(size=n)
        exact, _ = solve_obstacle(a, f, psi)
        w = psi + rng.exponential(size=n)
        lam = rng.exponential(size=n)
        sharp, gap, r, c = certificate(a, f, psi, w, lam)
        error = norm_a(w-exact, a)
        assert error <= sharp + 1e-9
        assert sharp <= gap + 1e-9
        primal = .5*w@a@w-f@w
        dual = -.5*(f+lam)@np.linalg.solve(a, f+lam)+lam@psi
        assert abs(primal-dual-(.5*r*r+c)) < 1e-8
        ratios.append(error/sharp)
    result["primal_dual_random_cases"] = 120
    result["largest_error_over_sharp_bound"] = max(ratios)
    equality_error = []
    for r in [0., .1, 1., 7.]:
        for c in [0., .2, 2., 11.]:
            t = phi(r, c)
            lam = t-r
            assert lam >= -1e-12
            equality_error.append(abs(t*t-r*t-c))
    result["sharpness_scalar_max_residual"] = max(equality_error)

    envelope_cases = []
    for trial in range(24):
        dims = (4, 5, 4)
        items = [fem_1d(n) for n in dims]
        ks, ms, bs = [list(t) for t in zip(*items)]
        bubble = separable_dense(bs)
        psi = -.2*bubble
        u = psi + bubble*(.1 + rng.uniform(size=dims))
        u_tt = tt_from_dense(u)
        weighted_u = tt_scale_separable(u_tt, [1/b for b in bs])
        v_scaled = tt_from_dense(tt_dense(weighted_u), max_rank=2)
        v_tt = tt_scale_separable(v_scaled, bs)
        t_tt = tt_add(weighted_u, v_scaled, sign=-1)
        delta = tt_norm(t_tt)
        ef = rank_one_envelope(t_tt)
        e_local = [b*e for b, e in zip(bs, ef)]
        local_tt = [b.reshape(1, -1, 1) for b in e_local]
        global_factors = [b.copy() for b in bs]
        global_factors[0] *= delta
        global_tt = [b.reshape(1, -1, 1) for b in global_factors]
        local_energy = tt_energy(local_tt, ks, ms)
        global_energy = tt_energy(global_tt, ks, ms)
        correction = local_tt if local_energy <= global_energy else global_tt
        w_tt = tt_add(v_tt, correction)
        w, v = tt_dense(w_tt), tt_dense(v_tt)
        error = u-v
        assert np.max(np.abs(error)-tt_dense(local_tt)) < 2e-10
        assert np.min(w-psi) > -2e-10
        assert all(w_tt[k].shape[2] <= v_tt[k].shape[2]+1 for k in range(2))
        input_error = tt_energy(tt_add(u_tt, v_tt, -1), ks, ms)
        repaired_error = tt_energy(tt_add(u_tt, w_tt, -1), ks, ms)
        selected_energy = min(local_energy, global_energy)
        assert repaired_error <= input_error+selected_energy+1e-8
        assert selected_energy <= global_energy+1e-12
        envelope_cases.append({"local_over_global_energy":local_energy/global_energy,
            "selected":"local" if local_energy<=global_energy else "global"})
    result["tt_envelope_cases"] = len(envelope_cases)
    result["local_envelope_selected"] = sum(t["selected"]=="local" for t in envelope_cases)
    result["local_over_global_energy_range"] = [min(t["local_over_global_energy"] for t in envelope_cases),
        max(t["local_over_global_energy"] for t in envelope_cases)]

    mesh = []
    for n in [7, 15, 31, 63, 127, 255]:
        k, m, b = fem_1d(n)
        h = 1/(n+1)
        v = np.zeros(n)
        v[0] = -h
        delta = h/b[0]
        w = v+delta*b
        err = norm_a(v, k)
        repaired = norm_a(w, k)
        assert np.min(w) >= -1e-12
        assert abs(norm_a(b, k)**2-(1-h*h)/3) < 1e-10
        assert abs(norm_a(np.ones(n), k)**2-2/h) < 1e-9
        exact_formula = delta*delta*norm_a(b, k)**2-4*delta*h*h+2*h
        assert abs(repaired*repaired-exact_formula) < 1e-10
        alternating = (-1.)**np.arange(n)
        assert norm_a(alternating, k) > norm_a(np.ones(n), k)
        mesh.append({"n":n,"h":h,"bubble_energy":norm_a(b,k),
            "constant_interior_lift_energy":norm_a(np.ones(n),k),
            "input_error":err,"repaired_error":repaired,"ratio":repaired/err})
    result["mesh_counterexample"] = mesh

    # Matrix-order TT obstruction: clipping can increase exact rank.
    n = 12
    ii = np.arange(n)
    x = ii[:, None]-ii[None, :]+.5
    assert np.linalg.matrix_rank(x) == 2
    assert np.linalg.matrix_rank(np.maximum(x, 0)) == n
    result["clipping_rank_example"] = {"before":2,"after":n}

    # Inexact proximal iterates with a residual-based relative stopping rule.
    n = 5
    z = rng.normal(size=(n,n))
    a = z.T@z+np.eye(n)
    m = np.diag(rng.uniform(.5, 2., size=n))
    invsqrt_m = np.diag(1/np.sqrt(np.diag(m)))
    coercivity = float(np.linalg.eigvalsh(invsqrt_m@a@invsqrt_m)[0])
    tau = 1/coercivity
    q = 1/(1+tau*coercivity)
    theta = (1-q)/4
    qhat = (q+theta)/(1-theta)
    psi, f = rng.normal(size=n), rng.normal(size=n)
    exact, _ = solve_obstacle(a, f, psi)
    x = psi+1
    anchor = psi+np.arange(1, n+1)/n
    errors = [norm_a(x-exact,m)]
    for _ in range(24):
        bmat = a+m/tau
        rhs = f+m@x/tau
        zz, lam = solve_obstacle(bmat,rhs,psi)
        alpha = .5
        for attempt in range(80):
            y = zz+alpha*(anchor-zz)
            bound, _, _, _ = certificate(bmat,rhs,psi,y,lam)
            eta = bound/np.sqrt(coercivity+1/tau)
            step = norm_a(y-x,m)
            if eta <= theta*step+1e-13:
                break
            alpha *= .5
        else:
            raise AssertionError("Inner audit did not meet relative tolerance")
        new_error = norm_a(y-exact,m)
        assert new_error <= qhat*errors[-1]+2e-10
        x = y
        errors.append(new_error)
    result["inexact_proximal"] = {"q":q,"theta":theta,"proven_qhat":qhat,
        "initial_error_M":errors[0],"final_error_M":errors[-1],"steps":24,
        "max_observed_ratio":max(y/x for x,y in zip(errors,errors[1:]) if x>1e-10)}
    result["status"] = "all numerical consistency checks passed; not interval certification"
    return result


if __name__ == "__main__":
    data = run_audits()
    target = Path(__file__).with_name("verification_results.json")
    target.write_text(json.dumps(data, indent=2)+"\n")
    print(json.dumps(data, indent=2))
