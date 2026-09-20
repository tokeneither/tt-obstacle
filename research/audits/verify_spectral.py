#!python
"""Small dense checks of spectral_envelope.md.

Floating-point sanity checks, not a scalable solver or interval certificates.
All prescribed nonzero singular directions are retained. Dense reference
matrices and full SVDs are intentional.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

OUTPUT = Path(__file__).with_name("spectral_verification_results.json")
RTOL = 5e-10


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def close(a, b, message):
    require(abs(a - b) <= RTOL * max(abs(a), abs(b), 1.0), message)


def bounded(a, b, message):
    require(a <= b + RTOL * max(abs(a), abs(b), 1.0), message)


def q1_matrices(n, rng):
    # Assemble INCLUDING boundary nodes, then restrict all three matrices.
    lengths = np.exp(rng.uniform(-2.5, 2.5, size=n + 1))
    lengths /= lengths.sum()
    k = np.zeros((n + 2, n + 2))
    m = np.zeros_like(k)
    d = np.zeros_like(k)
    for j, h in enumerate(lengths):
        ix = np.ix_([j, j + 1], [j, j + 1])
        k[ix] += np.array([[1., -1.], [-1., 1.]]) / h
        m[ix] += h * np.array([[2., 1.], [1., 2.]]) / 6
        d[ix] += h * np.eye(2) / 2
    return k[1:-1, 1:-1], m[1:-1, 1:-1], d[1:-1, 1:-1]


def z_matrix(n, rng):
    lap = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    scaling = np.exp(rng.uniform(-2., 2., size=n))
    return scaling[:, None] * lap * scaling[None, :]


def prescribed_matrix(n, m, spectrum, rng):
    s = len(spectrum)
    left = np.linalg.qr(rng.normal(size=(n, s)), mode="reduced")[0]
    right = np.linalg.qr(rng.normal(size=(m, s)), mode="reduced")[0]
    return (left * spectrum[None, :]) @ right.T


def spectral_envelope(r, dx, dy):
    sx, sy = np.sqrt(np.diag(dx)), np.sqrt(np.diag(dy))
    scaled = sx[:, None] * r * sy[None, :]
    left, sigma, right_t = np.linalg.svd(scaled, full_matrices=False)
    # Tests have full prescribed min(n,m) rank; include ALL singular values.
    require(np.all(sigma > 0), "Unexpected zero singular value")
    labels = np.floor(np.log2(sigma[0] / sigma)).astype(int)
    e = np.zeros_like(r)
    c_squared = 0.
    blocks = []
    for label in np.unique(labels):
        keep = np.flatnonzero(labels == label)
        sig = sigma[keep]
        us, vs = left[:, keep], right_t[keep, :].T
        a = np.sqrt((us * us) @ sig) / sx
        b = np.sqrt((vs * vs) @ sig) / sy
        eb = np.outer(a, b)
        rb = ((us * sig[None, :]) @ vs.T) / sx[:, None] / sy[None, :]
        cb_squared = float(sig.sum() / sig.min())
        e += eb
        c_squared += cb_squared
        blocks.append((rb, eb, cb_squared, len(keep), float(sig.max() / sig.min())))
    return e, sigma, blocks, float(c_squared)


def energy_trace(x, kx, ky, mx, my):
    return float(np.sum(x * (kx @ x @ my + mx @ x @ ky)))


def energy_dense(x, matrix):
    z = x.ravel(order="C")
    return float(z @ matrix @ z)


def run_case(name, kx, ky, mx, my, dx, dy, weighted_r, rng, is_q1):
    sx, sy = np.sqrt(np.diag(dx)), np.sqrt(np.diag(dy))
    r = weighted_r / sx[:, None] / sy[None, :]
    e, sigma, blocks, c_squared = spectral_envelope(r, dx, dy)
    s, bands = len(sigma), len(blocks)
    a_d = np.kron(kx, dy) + np.kron(dx, ky)
    a_fe = np.kron(kx, my) + np.kron(mx, ky)
    rd, ed = energy_dense(r, a_d), energy_dense(e, a_d)
    rf, ef = energy_dense(r, a_fe), energy_dense(e, a_fe)
    close(rd, energy_trace(r, kx, ky, dx, dy), name + ": lumped trace energy")
    close(rf, energy_trace(r, kx, ky, mx, my), name + ": FE trace energy")
    close(ef, energy_trace(e, kx, ky, mx, my), name + ": envelope trace energy")
    scale = max(float(np.max(np.abs(r))), float(np.max(e)), 1.)
    min_margin = float(np.min(e - np.abs(r)))
    require(min_margin >= -RTOL * scale, name + ": pointwise envelope")
    numerical_rank = int(np.linalg.matrix_rank(
        e, tol=1e-9 * max(np.linalg.norm(e, 2), 1.)))
    require(numerical_rank <= bands, name + ": rank exceeds number of bands")
    rank_bound = min(s, 1 + int(np.floor(np.log2(sigma[0] / sigma[-1]))))
    require(bands <= rank_bound, name + ": logarithmic rank bound")
    bounded(c_squared, 2. * s, name + ": dyadic constant")
    bounded(ed, c_squared * rd, name + ": sharper lumped C energy bound")
    bounded(ef, 3. * c_squared * rf, name + ": sharper FE sqrt(3) C bound")
    bounded(ef, 6. * s * rf, name + ": FE sqrt(6s) bound")
    block_energy_sum = 0.
    for rb, eb, cb_squared, size, kappa in blocks:
        rb_energy = energy_dense(rb, a_d)
        bounded(energy_dense(eb, a_d), cb_squared * rb_energy,
                name + ": block energy bound")
        bounded(cb_squared, size * kappa, name + ": block size*kappa")
        bounded(kappa, 2., name + ": dyadic block condition")
        block_energy_sum += rb_energy
    close(block_energy_sum, rd, name + ": block energy Pythagoras")

    psi = rng.normal(size=r.shape)
    slack = np.abs(rng.normal(size=r.shape))
    slack[::2, ::2] = 0.  # Include actual contact nodes.
    u = psi + slack
    v, w = u - r, u - r + e
    min_repair_margin = float(np.min(w - u))
    require(min_repair_margin >= -RTOL * scale, name + ": W >= U")
    require(float(np.min(w - psi)) >= -RTOL * scale, name + ": obstacle")
    repair_constant = 1. + np.sqrt(3. * c_squared)
    bounded(energy_dense(w - u, a_fe), repair_constant**2 * rf,
            name + ": repair energy bound")
    # Rank of repaired sum, using a separate dense calculation.
    rank_v = int(np.linalg.matrix_rank(v))
    require(np.linalg.matrix_rank(w) <= rank_v + bands, name + ": repaired rank")

    if is_q1:
        lower = float(np.linalg.eigvalsh(a_fe - a_d / 3.)[0])
        upper = float(np.linalg.eigvalsh(a_d - a_fe)[0])
        matrix_scale = max(float(np.linalg.norm(a_d, 2)), 1.)
        require(lower >= -RTOL * matrix_scale, name + ": FE >= lumped/3")
        require(upper >= -RTOL * matrix_scale, name + ": FE <= lumped")
    else:
        lower = upper = None
    return {
        "name": name, "shape": list(r.shape), "residual_rank": s,
        "weighted_singular_condition": float(sigma[0] / sigma[-1]),
        "bands": bands, "numerical_envelope_rank": numerical_rank,
        "C_squared": c_squared,
        "envelope_to_residual_lumped_energy_norm": float(np.sqrt(ed / rd)),
        "envelope_to_residual_FE_energy_norm": float(np.sqrt(ef / rf)),
        "sharper_lumped_bound_fraction": float(ed / (c_squared * rd)),
        "sharper_FE_bound_fraction": float(ef / (3. * c_squared * rf)),
        "sqrt_6s_FE_bound_fraction": float(ef / (6. * s * rf)),
        "min_envelope_margin": min_margin, "min_repair_margin": min_repair_margin,
        "min_eigenvalue_FE_minus_lumped_over_3": lower,
        "min_eigenvalue_lumped_minus_FE": upper,
        "minimum_modular_row_sum": float(min(kx.sum(axis=1).min(),
                                             ky.sum(axis=1).min())),
    }


def spectra(s):
    return {
        "flat": np.ones(s),
        "dyadic": 2. ** (-np.arange(s, dtype=float)),
        "clustered": np.array([1., .92, .77, .38, .31, .18, .08, .065, .025, .012])[:s],
        "spread": np.geomspace(1., 1e-7, s),
    }


def main():
    records = []
    for seed in range(4):
        rng = np.random.default_rng(9100 + seed)
        for n, m in [(4, 5), (7, 6), (10, 9)]:
            kx, mx, dx = q1_matrices(n, rng)
            ky, my, dy = q1_matrices(m, rng)
            for spectrum_name, sigma in spectra(min(n, m)).items():
                wr = prescribed_matrix(n, m, sigma, rng)
                records.append(run_case(
                    f"nonuniform_Q1_seed{seed}_{n}x{m}_{spectrum_name}",
                    kx, ky, mx, my, dx, dy, wr, rng, True))
    for seed in range(8):
        rng = np.random.default_rng(10200 + seed)
        n, m = 6, 7
        kx, ky = z_matrix(n, rng), z_matrix(m, rng)
        require(min(kx.sum(axis=1).min(), ky.sum(axis=1).min()) < 0,
                "Z example must test negative row sums")
        wr = prescribed_matrix(n, m, np.geomspace(1., .004, n), rng)
        records.append(run_case(
            f"SPD_Z_negative_rows_seed{seed}", kx, ky,
            np.eye(n), np.eye(m), np.eye(n), np.eye(m), wr, rng, False))

    equality_cases = []
    for s in [2, 4, 8, 12]:
        r = np.eye(s)
        e, sigma, blocks, c_squared = spectral_envelope(r, np.eye(s), np.eye(s))
        close(float(np.max(np.abs(e - np.ones_like(e)))), 0., "Identity envelope")
        require(len(blocks) == 1, "Constant spectrum has one band")
        close(float(np.linalg.norm(e) / np.linalg.norm(r)), np.sqrt(s),
              "Rank-one sharp equality")
        close(c_squared, float(s), "Sharp block constant")
        for k in [k for k in [1, 2, 3, 4, 6, 12] if k <= s and s % k == 0]:
            ek = np.kron(np.eye(k), np.ones((s // k, s // k)))
            require(np.all(ek >= r), "Block envelope dominates identity")
            require(np.linalg.matrix_rank(ek) == k, "Block-envelope rank")
            ratio = float(np.linalg.norm(ek) / np.linalg.norm(r))
            close(ratio, np.sqrt(s / k), "General rank-energy equality")
            equality_cases.append({"s": s, "k": k, "energy_norm_ratio": ratio,
                                   "lower_bound": float(np.sqrt(s / k))})
    # An omitted small singular tail can destroy the entrywise guarantee.
    omitted_r = np.diag([1., .1, .01])
    omitted_e = np.diag([1., .1, 0.])
    omitted_margin = float(np.min(omitted_e - np.abs(omitted_r)))
    require(omitted_margin < 0, "Omitted-tail example should fail dominance")

    results = {
        "status": "passed",
        "scope": "Small dense floating-point checks; not a scalable solver or interval certificate.",
        "tolerance": RTOL,
        "singular_directions": "All prescribed nonzero directions retained; no truncation threshold.",
        "checks": [
            "Nonuniform P1 stiffness, consistent mass and full-assembly lumped mass",
            "Dense Kronecker energy agrees with matrix-trace reference",
            "Entrywise domination and W >= U >= obstacle including contact nodes",
            "Envelope rank bounded by nonempty dyadic bands and logarithmic bound",
            "Block energy bounds and SVD-block energy Pythagoras",
            "Sharper C lumped-energy and sqrt(3)*C consistent-energy bounds",
            "sqrt(6s) bound for consistent Q1 Galerkin energy",
            "SPD Z-matrices with negative row sums",
            "Sharp identity examples for rank-one and rank-k lower bounds",
            "Discarded singular tail deliberately violates domination"],
        "random_case_count": len(records),
        "sharp_equality_case_count": len(equality_cases),
        "max_sharper_lumped_bound_fraction":
            max(r["sharper_lumped_bound_fraction"] for r in records),
        "max_sharper_FE_bound_fraction":
            max(r["sharper_FE_bound_fraction"] for r in records),
        "max_sqrt_6s_FE_bound_fraction":
            max(r["sqrt_6s_FE_bound_fraction"] for r in records),
        "minimum_pointwise_margin": min(r["min_envelope_margin"] for r in records),
        "omitted_tail_margin_expected_negative": omitted_margin,
        "random_cases": records, "sharp_equality_cases": equality_cases}
    OUTPUT.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items()
                      if k not in {"random_cases", "sharp_equality_cases"}}, indent=2))


if __name__ == "__main__":
    main()
