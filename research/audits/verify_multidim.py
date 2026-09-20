"""Numerical audits of the multidimensional proofs (not interval certification).

Large-d additive examples use TT cores of size at most 3 and contractions only.
Small dense references audit the tensor contractions and entrywise guarantees.
Run: python verify_multidim.py
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import numpy as np

from verify_theory import tt_dense, tt_product_inner


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def relative_close(x, y, message, tol=3e-8):
    require(abs(x-y) <= tol*max(abs(x), abs(y), 1e-100), message)


def assemble(lengths):
    n = len(lengths)-1
    k = np.zeros((n+2, n+2))
    m = np.zeros_like(k)
    for j, h in enumerate(lengths):
        ix = np.ix_([j, j+1], [j, j+1])
        k[ix] += np.array([[1., -1.], [-1., 1.]])/h
        m[ix] += h*np.array([[2., 1.], [1., 2.]])/6
    return k[1:-1, 1:-1], m[1:-1, 1:-1]


def eigenbasis(k, m):
    chol = np.linalg.cholesky(m)
    inv = np.linalg.inv(chol)
    values, vectors = np.linalg.eigh(inv@k@inv.T)
    vectors = np.linalg.solve(chol.T, vectors)
    if vectors[:, 0].sum() < 0:
        vectors[:, 0] *= -1
    require(np.all(vectors[:, 0] > 0), "Ground state not positive")
    return values, vectors


def additive_cores(bs, ts):
    d = len(bs)
    require(d >= 2, "At least two modes for this diagnostic")
    cores = [np.stack([ts[0], np.ones_like(ts[0])], axis=1)[None, :, :]]
    for t in ts[1:-1]:
        g = np.zeros((2, len(t), 2))
        g[0, :, 0] = g[1, :, 1] = 1
        g[1, :, 0] = t
        cores.append(g)
    cores.append(np.stack([np.ones_like(ts[-1]), ts[-1]], axis=0)[:, :, None])
    return [g*b[None, :, None] for g, b in zip(cores, bs)]


def quadratic_cores(bs, ts, constant):
    """B*((sum t_k)^2+constant), using three states 1, S, S^2."""
    first = np.stack([np.ones_like(ts[0]), ts[0], ts[0]**2], axis=1)
    cores = [first[None, :, :]]
    for t in ts[1:-1]:
        g = np.zeros((3, len(t), 3))
        g[0, :, 0] = g[1, :, 1] = g[2, :, 2] = 1
        g[0, :, 1] = t
        g[0, :, 2] = t*t
        g[1, :, 2] = 2*t
        cores.append(g)
    last = np.stack([ts[-1]**2+constant, 2*ts[-1], np.ones_like(ts[-1])])
    cores.append(last[:, :, None])
    return [g*b[None, :, None] for g, b in zip(cores, bs)]


def tensor_energy(cores, ks, ms):
    # Simultaneously accumulate the tensor mass and the sum with one K factor.
    mass = np.ones((1, 1))
    energy = np.zeros((1, 1))
    for g, k, m in zip(cores, ks, ms):
        def advance(z, a):
            return np.einsum('ab,aic,ij,bjd->cd', z, g, a, g, optimize=True)
        energy = advance(energy, m)+advance(mass, k)
        mass = advance(mass, m)
    return float(energy[0, 0]), float(mass[0, 0])


def full_operator(ks, ms):
    n = int(np.prod([len(k) for k in ks]))
    a = np.zeros((n, n))
    for axis in range(len(ks)):
        term = np.ones((1, 1))
        for j in range(len(ks)):
            term = np.kron(term, ks[j] if j == axis else ms[j])
        a += term
    return a


def run_additive(d, n, nonuniform, rng):
    ks, ms, bs, ts = [], [], [], []
    variances, dirichlets, fvars, fdirs, fmeans, eigenvalues, peaks = ([] for _ in range(7))
    for axis in range(d):
        lengths = (np.exp(rng.uniform(-1.7, 1.7, size=n+1))
                   if nonuniform else np.ones(n+1))
        lengths /= lengths.sum()
        k, m = assemble(lengths)
        values, vectors = eigenbasis(k, m)
        b = vectors[:, 0]
        if nonuniform:
            coefficients = rng.normal(size=n-1)
            r = vectors[:, 1:]@coefficients
        else:
            r = rng.normal()*vectors[:, 1]
        v = float(r@m@r)
        t = r/b
        f = r*r/b
        mean = float(b@m@f)
        shifted = k-values[0]*m
        g = f-mean*b
        ks.append(k); ms.append(m); bs.append(b); ts.append(t)
        variances.append(v); dirichlets.append(float(r@shifted@r))
        fvars.append(float(g@m@g)); fdirs.append(float(f@shifted@f))
        fmeans.append(mean); eigenvalues.append(float(values[0]))
        peaks.append(float(np.max(np.abs(t))/np.sqrt(v)))
        require(abs(b@m@r) < 1e-8*max(1., np.sqrt(v)), "Centering failed")
    variance = sum(variances)
    ground_eigenvalue = sum(eigenvalues)
    kappa = max(1., max(peaks))
    if not nonuniform:
        require(kappa <= 2+1e-9, "Uniform sine subclass kappa > 2")
    rcores = additive_cores(bs, ts)
    fcores = quadratic_cores(bs, ts, 0.)
    ecores = quadratic_cores(bs, ts, variance)
    ecores[-1] /= 2*np.sqrt(variance)
    renergy, rmass = tensor_energy(rcores, ks, ms)
    fenergy, fmass = tensor_energy(fcores, ks, ms)
    eenergy, emass = tensor_energy(ecores, ks, ms)
    predicted_mass_f = sum(fmeans)**2+sum(fvars)+2*(variance**2-sum(v*v for v in variances))
    predicted_dir_f = sum(fdirs)+4*sum((variance-v)*dv for v, dv in zip(variances, dirichlets))
    relative_close(rmass, variance, "R mass identity")
    relative_close(renergy, ground_eigenvalue*variance+sum(dirichlets), "R energy identity")
    relative_close(fmass, predicted_mass_f, "F mass identity")
    relative_close(fenergy, ground_eigenvalue*predicted_mass_f+predicted_dir_f, "F energy identity")
    predicted_e = fenergy/(4*variance)+ground_eigenvalue*sum(fmeans)/2+ground_eigenvalue*variance/4
    relative_close(eenergy, predicted_e, "Envelope energy identity")
    bound_squared = (kappa**4+2*kappa*kappa+3)/4
    require(eenergy <= bound_squared*renergy*(1+1e-8), "Dimension-free FE energy bound")
    require(max(g.shape[2] for g in ecores) <= 3, "Rank-three representation")
    if d <= 4 and n <= 4:
        rd, ed, fd = tt_dense(rcores), tt_dense(ecores), tt_dense(fcores)
        a = full_operator(ks, ms)
        relative_close(renergy, float(rd.ravel()@a@rd.ravel()), "Dense R energy")
        relative_close(eenergy, float(ed.ravel()@a@ed.ravel()), "Dense E energy")
        require(np.min(ed-np.abs(rd)) > -1e-9, "Dense envelope domination")
        bd = tt_dense([b[None, :, None] for b in bs])
        require(np.allclose(fd, rd*rd/bd), "Dense polynomial identity")
    # Scalar evaluations audit entrywise guarantees without creating a full array.
    for _ in range(100):
        indices = [int(rng.integers(n)) for _ in range(d)]
        scalar_t = sum(t[i] for t, i in zip(ts, indices))
        require((scalar_t**2+variance)/(2*np.sqrt(variance)) >= abs(scalar_t)-1e-10,
                "Scalar polynomial domination")
    return {"d": d, "n": n, "nonuniform_mesh": nonuniform,
            "kappa": kappa, "energy_amplification": float(np.sqrt(eenergy/renergy)),
            "proven_amplification": float(np.sqrt(bound_squared)),
            "input_TT_core_rank": 2, "envelope_TT_core_rank": 3,
            "full_entries_not_assembled": str(n**d)}


def run_no_go():
    k = 3*np.array([[2., -1.], [-1., 2.]])
    m = np.array([[4., 1.], [1., 4.]])/18
    b, z = np.ones(2), np.array([1., -1.])
    cases = []
    for d in [2, 4, 8, 16, 32, 64]:
        be, _ = tensor_energy([b[None, :, None]]*d, [k]*d, [m]*d)
        re, _ = tensor_energy([z[None, :, None]]*d, [k]*d, [m]*d)
        ratio2 = (5/3)**(d-1)/3
        relative_close(be/re, ratio2, "Q1 checkerboard exact ratio")
        if d <= 8:
            a = full_operator([k]*d, [m]*d)
            bd = np.ones(2**d)
            rd = tt_dense([z[None, :, None]]*d).ravel()
            require(np.min(a@bd) > 0, "KKT envelope lower bound")
            if d >= 4:
                w = np.maximum(-rd, 0.)
                lam = a@w
                require(np.min(lam) > -1e-10, "Positive-repair KKT multiplier")
                relative_close(float(w@a@w)/re, (1+ratio2)/4, "Positive-repair exact bound")
        cases.append({"d": d, "absolute_envelope_min_ratio": float(np.sqrt(ratio2)),
                      "positive_repair_min_error_ratio": float(.5*np.sqrt(1+ratio2)) if d>=4 else None})
    return cases


def boolean_energy(x, weights, reaction):
    e = reaction*np.mean(x*x)
    for j, weight in enumerate(weights):
        e += .5*weight*np.mean((x-np.flip(x, axis=j))**2)
    return float(e)


def run_boolean(rng):
    cases = []
    for d in [4, 6, 8]:
        points = np.array(list(itertools.product([-1., 1.], repeat=d)))
        for degree in [1, 2, 3]:
            t = np.zeros(len(points))
            coeffs = []
            for size in range(degree+1):
                for subset in itertools.combinations(range(d), size):
                    coefficient = rng.normal()/np.sqrt(d**size)
                    t += coefficient*np.prod(points[:, subset], axis=1)
                    coeffs.append((subset, coefficient))
            t = t.reshape((2,)*d)
            sigma2 = float(np.mean(t*t))
            env = (t*t+sigma2)/(2*np.sqrt(sigma2))
            weights = np.exp(rng.uniform(-1, 1, size=d))
            reaction = 1.7
            energy = boolean_energy(t, weights, reaction)
            spectral_energy = sum((reaction+2*sum(weights[j] for j in subset))*c*c
                                  for subset, c in coeffs)
            relative_close(energy, spectral_energy, "Boolean energy independent Fourier check")
            env_energy = boolean_energy(env, weights, reaction)
            bound = 3**(degree-.5)
            require(env_energy <= bound*bound*energy*(1+1e-10), "Boolean bounded degree bound")
            require(np.min(env-np.abs(t)) >= -1e-10, "Boolean domination")
            for cut in range(1, d):
                rank_r = np.linalg.matrix_rank(t.reshape(2**cut, -1), tol=1e-9)
                rank_e = np.linalg.matrix_rank(env.reshape(2**cut, -1), tol=1e-9)
                require(rank_e <= rank_r*(rank_r+1)//2+1, "Symmetric-square rank bound")
            cases.append({"d":d, "interaction_degree":degree,
                          "energy_amplification":float(np.sqrt(env_energy/energy)),
                          "proven_amplification":bound})
    return cases


def main():
    rng = np.random.default_rng(20260921)
    additive = [run_additive(d, n, False, rng) for d, n in
                [(2,3),(3,4),(4,3),(8,7),(16,7),(32,7),(64,7),(8,15),(32,15)]]
    additive += [run_additive(d, n, True, rng) for d, n in
                 [(2,3),(3,4),(4,3),(8,5),(16,5),(32,5)]]
    results = {"status":"passed", "scope":"Floating-point audits, not interval certification or a PDE solver",
               "additive_Q1_cases":additive, "Q1_no_go":run_no_go(),
               "Boolean_low_interaction_cases":run_boolean(rng)}
    Path(__file__).with_name('multidim_verification_results.json').write_text(json.dumps(results, indent=2))
    print(json.dumps({"status":results["status"], "additive_Q1_cases":len(additive),
                      "max_d":max(x['d'] for x in additive), "max_envelope_core_rank":3,
                      "Boolean_cases":len(results['Boolean_low_interaction_cases']),
                      "no_go_d64_amplification":results['Q1_no_go'][-1]['absolute_envelope_min_ratio']}, indent=2))


if __name__ == '__main__':
    main()
